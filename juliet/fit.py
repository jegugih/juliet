# juliet's backend is written in JAX: lightcurves and radial-velocities are computed with jaxoplanet,
# Gaussian Processes with celerite2's kernels and a JAX celerite solver (or dense JAX GPs for the george-like kernels), and the
# posteriors are sampled with JAX samplers (blackjax's nested slice sampling or numpyro's MCMCs).
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import sys
import copy
import time
import pickle
import inspect
import numpy as np

from numpyro.distributions.transforms import biject_to

from . import jaxmodels as jm
from .jaxgp import JaxGP, kernel_variables
from .samplers import run_nested, run_numpyro, chunked_vmap

# Define constants on the code:
G = 6.67408e-11  # Gravitational constant, mks
log2pi = np.log(2. * np.pi)  # ln(2*pi)

# Import all the utils functions:
from .utils import *

__all__ = ['load', 'fit', 'gaussian_process', 'model']

class load(object):
    """
    Given a dictionary with priors (or a filename pointing to a prior file) and data either given through arrays
    or through files containing the data, this class loads data into a juliet object which holds all the information
    about the dataset. Example usage:

               >>> data = juliet.load(priors=priors,t_lc=times,y_lc=fluxes,yerr_lc=fluxes_errors)

    Or, also,

               >>> data = juliet.load(input_folder = folder)

    :param priors: (optional, dict or string)
        This can be either a python ``string`` or a python ``dict``. If a ``dict``, this has to contain each of
        the parameters to be fit, along with their respective prior distributions and hyperparameters. Each key
        of this dictionary has to have a parameter name (e.g., ``r1_p1``, ``sigma_w_TESS``), and each of
        those elements are, in turn, dictionaries as well containing two keys: a ``distribution``
        key which defines the prior distribution of the parameter and a ``hyperparameters`` key,
        which contains the hyperparameters of that distribution.

        Example setup of the ``priors`` dictionary:
            >>> priors = {}
            >>> priors['r1_p1'] = {}
            >>> priors['r1_p1']['distribution'] = 'Uniform'
            >>> priors['r1_p1']['hyperparameters'] = [0.,1.]

        If a ``string``, this has to contain the filename to a proper juliet prior file; the prior ``dict`` will
        then be generated from there. A proper prior file has in the first column the name of the parameter,
        in the second the name of the distribution, and in the third the hyperparameters of that distribution for
        the parameter.

        Note that this along with either lightcurve or RV data or a ``input_folder`` has to be given in order to properly
        load a juliet data object.

    :param starting_point: (mandatory if using MCMC, useless if using nested samplers, dict)
        Dictionary indicating the starting value of each of the parameters for the MCMC run (i.e., currently only of use for ``emcee``). Keys should be consistent with the ``prior`` namings above;
        each key should have an associated float with the starting value. This is of no use if using nested samplers (which sample directly from the prior).

    :param input_folder: (optional, string)
        Python ``string`` containing the path to a folder containing all the input data --- this will thus be load into a
        juliet data object. This input folder has to contain at least a ``priors.dat`` file with the priors and either a ``lc.dat``
        file containing lightcurve data or a ``rvs.dat`` file containing radial-velocity data. If in this folder a ``GP_lc_regressors.dat``
        file or a ``GP_rv_regressors.dat`` file is found, data will be loaded into the juliet object as well.

        Note that at least this or a ``priors`` string or dictionary, along with either lightcurve or RV data has to be given
        in order to properly load a juliet data object.

    :param t_lc: (optional, dictionary)
        Dictionary whose keys are instrument names; each of those keys is expected to have arrays with the times corresponding to those instruments.
        For example,
                                    >>> t_lc = {}
                                    >>> t_lc['TESS'] = np.linspace(0,100,100)

        Is a valid input dictionary for ``t_lc``.

    :param y_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the fluxes corresponding to those instruments.
        These are expected to be consistent with the ``t_lc`` dictionaries.

    :param yerr_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the errors on the fluxes corresponding to those instruments.
        These are expected to be consistent with the ``t_lc`` dictionaries.

    :param GP_regressors_lc: (optional, dictionary)
        Dictionary whose keys are names of instruments where a GP is to be fit. On each name/element, an array of
        regressors of shape ``(m,n)`` containing in each column the ``n`` GP regressors to be used for
        ``m`` photometric measurements has to be given. Note that ``m`` for a given instrument has to be of the same length
        as the corresponding ``t_lc`` for that instrument. Also, note the order of each regressor of each instrument has to match
        the corresponding order in the ``t_lc`` array.
        For example,

                                    >>> GP_regressors_lc = {}
                                    >>> GP_regressors_lc['TESS'] = np.linspace(-1,1,100)

        If a global model wants to be used, then the instrument should be ``rv``, and each of the ``m`` rows should correspond to the ``m`` times.

    :param linear_regressors_lc: (optional, dictionary)
        Similarly as for ``GP_regressors_lc``, this is a dictionary whose keys are names of instruments where a linear regression is to be fit.
        On each name/element, an array of shape ``(q,p)`` containing in each column the ``p`` linear regressors to be used for the ``q``
        photometric measurements. Again, note the order of each regressor of each instrument has to match the corresponding order in the ``t_lc`` array.

    :param GP_regressors_rv: (optional, dictionary)
        Same as ``GP_regressors_lc`` but for the radial-velocity data. If a global model wants to be used, then the instrument should be ``lc``, and each of the ``m`` rows should correspond to the ``m`` times.

    :param linear_regressors_rv: (optional, dictionary)
        Same as ``linear_regressors_lc``, but for the radial-velocities.

    :param t_rv: (optional, dictionary)
        Same as ``t_lc``, but for the radial-velocities.

    :param y_rv: (optional, dictionary)
        Same as ``y_lc``, but for the radial-velocities.

    :param yerr_rv: (optional, dictionary)
        Same as ``yerr_lc``, but for the radial-velocities.

    :param out_folder: (optional, string)
        If a path is given, results will be saved to that path as a ``pickle`` file, along with all inputs in the standard juliet format.

    :param lcfilename:  (optional, string)
        If a path to a lightcurve file is given, ``t_lc``, ``y_lc``, ``yerr_lc`` and ``instruments_lc`` will be read from there. The basic file format is a pure
        ascii file where times are in the first column, relative fluxes in the second, errors in the third and instrument names in the fourth. If more columns are given for
        a given instrument, those will be identified as linear regressors for those instruments.

    :param rvfilename: (optional, string)
        Same as ``lcfilename``, but for the radial-velocities.

    :param GPlceparamfile: (optional, string)
        If a path to a file is given, the columns of that file will be used as GP regressors for the lightcurve fit. The file format is a pure ascii file
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with
        ``t_lc`` and/or the ``lcfilename`` file. If a global model wants to be used, set the instrument names of all regressors to ``lc``.

    :param GPrveparamfile: (optional, string)
        Same as ``GPlceparamfile`` but for the radial-velocities. If a global model wants to be used, set the instrument names of all regressors to ``rv``.

    :param LMlceparamfile: (optional, string)
        If a path to a file is given, the columns of that file will be used as linear regressors for the lightcurve fit. The file format is a pure ascii file
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with
        ``t_lc`` and/or the ``lcfilename`` file. If a global model wants to be used, set the instrument names of all regressors to ``lc``.

    :param LMrveparamfile: (optional, string)
        Same as ``LMlceparamfile`` but for the radial-velocities. If a global model wants to be used, set the instrument names of all regressors to ``rv``.

    :param lctimedef: (optional, string)
        Time definitions for each of the lightcurve instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given, this string
        should have instruments and time-definitions separated by commas, e.g., ``TESS-TDB, LCOGT-UTC``, etc.

    :param rvtimedef: (optional, string)
        Time definitions for each of the radial-velocity instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given,
        this string should have instruments and time-definitions separated by commas, e.g., ``FEROS-TDB, HARPS-UTC``, etc.

    :param ld_laws: (optional, string)
        Limb-darkening law to be used for each instrument. Default is ``quadratic`` for all instruments. If more than one instrument is given,
        this string should have instruments and limb-darkening laws separated by commas, e.g., ``TESS-quadratic, LCOGT-linear``.

    :param priorfile: (optional, string)
        If a path to a file is given, it will be assumed this is a prior file. The ``priors`` dictionary will be overwritten by the data in this
        file. The file structure is a plain ascii file, with the name of the parameters in the first column, name of the prior distribution in the
        second column and hyperparameters in the third column.

    :param lc_instrument_supersamp: (optional, array of strings)
        Define for which lightcurve instruments super-sampling will be applied (e.g., in the case of long-cadence integrations). e.g., ``lc_instrument_supersamp = ['TESS','K2']``

    :param lc_n_supersamp: (optional, array of ints)
        Define the number of datapoints to supersample. Order should be consistent with order in ``lc_instrument_supersamp``. e.g., ``lc_n_supersamp = [20,30]``.

    :param lc_exptime_supersamp: (optional, array of floats)
        Define the exposure-time of the observations for the supersampling. Order should be consistent with order in ``lc_instrument_supersamp``. e.g., ``lc_exptime_supersamp = [0.020434,0.020434]``

    :param verbose: (optional, boolean)
        If True, all outputs of the code are printed to terminal. Default is False.

    :param matern_eps: (optional, float)
        Epsilon parameter for the Matern approximation (see celerite documentation).

    :param george_hodlr: (optional, bool)
        Flag to define if you want to use the HODLR solver for george GP's or not (see http://dfm.io/george/current/user/solvers/).

    :param pickle_encoding: (optional, string)
        Define pickle encoding in case fit was done with Python 2.7 and results are read with Python 3.

    :param non_linear_functions: (optional, dict)
        Dictionary containing any non-linear functions (`non_linear_functions['function']`) and regressors (`non_linear_functions['regressor']`) that want to be fit.

    :param backend: (optional, string)
        ``jax`` (default) to compute models with jaxoplanet/celerite2 and sample with JAX samplers, or ``legacy`` to use the
        original implementation of juliet (batman, catwoman, radvel, george, celerite; MultiNest, dynesty, UltraNest, emcee
        and zeus as samplers), which requires those packages to be installed.

    :param legacy_gp_parametrization: (optional, boolean)
        If True, GP kernels use the parametrization of juliet <= 2.2.10, in which the exp-sine-squared kernel used
        ``log(GP_Gamma)`` (instead of ``GP_Gamma``) as the amplitude of its sine part, and the multi-dimensional
        squared-exponential and Matern 3/2 kernels had a variance of ``nX * GP_sigma**2`` for ``nX`` regressors (instead
        of ``GP_sigma**2``). Default is False for the ``jax`` backend and True for the ``legacy`` backend.

    """

    def __new__(cls, *args, backend = 'jax', **kwargs):
        if backend == 'legacy':
            from .legacy.fit import load as legacy_load
            return legacy_load(*args, **kwargs)
        elif backend != 'jax':
            raise Exception('INPUT ERROR: backend "' + str(backend) + '" not recognized; options are "jax" and "legacy".')
        return super().__new__(cls)

    def data_preparation(self, times, instruments, linear_regressors, non_linear_functions):
        """
        This function generates f useful internal arrays for this class: inames which saves the instrument names, ``global_times``
        which is a "flattened" array of the ``times`` dictionary where all the times for all instruments are stacked, instrument_indexes,
        which is a dictionary that has, for each instrument the indexes of the ``global_times`` corresponding to each instrument, lm_boolean which saves booleans for each
        instrument to indicate if there are linear regressors and lm_arguments which are the linear-regressors for each instrument.
        """
        inames = []
        for i in range(len(times)):
            if instruments[i] not in inames:
                inames.append(instruments[i])
        ninstruments = len(inames)
        instrument_indexes = {}
        for instrument in inames:
            instrument_indexes[instrument] = np.where(
                instruments == instrument)[0]

        # Also generate lm_lc_boolean in case linear regressors were passed:
        lm_boolean = {}
        lm_arguments = {}
        if linear_regressors is not None:
            linear_instruments = linear_regressors.keys()
            for instrument in inames:
                if instrument in linear_instruments:
                    lm_boolean[instrument] = True
                else:
                    lm_boolean[instrument] = False
        else:
            for instrument in inames:
                lm_boolean[instrument] = False

        nlm_boolean = {}
        for instrument in inames:

            if instrument in list( non_linear_functions.keys() ):

                nlm_boolean[instrument] = True

            else:

                nlm_boolean[instrument] = False            

        return inames, instrument_indexes, lm_boolean, nlm_boolean

    def convert_input_data(self, t, y, yerr):
        """
        This converts the input dictionaries to arrays (this is easier to handle internally within juliet; input dictionaries are just asked because
        it is easier for the user to pass them).
        """
        instruments = list(t.keys())
        all_times = np.array([])
        all_y = np.array([])
        all_yerr = np.array([])
        all_instruments = np.array([])
        for instrument in instruments:
            all_times = np.hstack(( all_times, t[instrument] ))
            all_y = np.hstack(( all_y, y[instrument] ))
            all_yerr = np.hstack(( all_yerr, yerr[instrument] ))
            all_instruments = np.hstack(( all_instruments, np.repeat(instrument, len(t[instrument]) ) ))
        return all_times, all_y, all_yerr, all_instruments

    def convert_to_dictionary(self, t, y, yerr, instrument_indexes):
        """
        Convert data given in arrays to dictionaries for easier user usage
        """
        times = {}
        data = {}
        errors = {}
        for instrument in instrument_indexes.keys():
            times[instrument] = t[instrument_indexes[instrument]]
            data[instrument] = y[instrument_indexes[instrument]]
            errors[instrument] = yerr[instrument_indexes[instrument]]
        return times, data, errors

    def save_regressors(self, fname, GP_arguments):
        """
        This function saves the GP regressors to fname.
        """
        fout = open(fname, 'w')
        for GP_instrument in GP_arguments.keys():
            GP_regressors = GP_arguments[GP_instrument]
            multi_dimensional = False
            if len(GP_regressors.shape) == 2:
                multi_dimensional = True
            if multi_dimensional:
                for i in range(GP_regressors.shape[0]):
                    for j in range(GP_regressors.shape[1]):
                        fout.write('{0:.10f} '.format(GP_regressors[i, j]))
                    fout.write('{0:}\n'.format(GP_instrument))
            else:
                for i in range(GP_regressors.shape[0]):
                    fout.write('{0:.10f} {1:}\n'.format(GP_regressors[i],
                                                        GP_instrument))
        fout.close()

    def save_data(self, fname, t, y, yerr, instruments, lm_boolean,
                  lm_arguments):
        """
        This function saves t,y,yerr,instruments,lm_boolean and lm_arguments data to fname.
        """
        fout = open(fname, 'w')
        lm_counters = {}
        for i in range(len(t)):

            fout.write('{0:.10f} {1:.10f} {2:.10f} {3:}'.format(
                t[i], y[i], yerr[i], instruments[i]))

            if lm_boolean[instruments[i]]:

                if instruments[i] not in lm_counters.keys():

                    lm_counters[instruments[i]] = 0

                for j in range(lm_arguments[instruments[i]].shape[1]):

                    fout.write(' {0:.10f}'.format(lm_arguments[instruments[i]][

                        lm_counters[instruments[i]]][j]))

                lm_counters[instruments[i]] += 1

            fout.write('\n')

        fout.close()

    def save_priorfile(self, fname):
        """
        This function saves a priorfile file out to fname
        """
        fout = open(fname, 'w')
        for pname in self.priors.keys():
            if self.priors[pname]['distribution'].lower() != 'fixed':
                value = ','.join(
                    np.array(self.priors[pname]['hyperparameters']).astype(str))
                if self.starting_point is not None:
                    fout.write('{0: <20} {1: <20} {2: <20} {3: <20}\n'.format(
                        pname, self.priors[pname]['distribution'], value,
                        self.starting_point[pname]))
                else:
                    fout.write('{0: <20} {1: <20} {2: <20}\n'.format(
                        pname, self.priors[pname]['distribution'], value))
            else:
                value = str(self.priors[pname]['hyperparameters'])
                fout.write('{0: <20} {1: <20} {2: <20}\n'.format(
                    pname, self.priors[pname]['distribution'], value))
        fout.close()

    def check_global(self, name):
        for pname in self.priors.keys():
            if name in pname.split('_')[1:]:
                return True
        return False

    def append_GP(self, ndata, instrument_indexes, GP_arguments, inames):
        """
            This function appends all the GP regressors into one --- useful for the global models.
        """
        # First check if GP regressors are multi-dimensional --- check this just for the first instrument:
        if len(GP_arguments[inames[0]].shape) == 2:
            nregressors = GP_arguments[inames[0]].shape[1]
            multidimensional = True
            out = np.zeros([ndata, nregressors])
        else:
            multidimensional = False
            out = np.zeros(ndata)
        for instrument in inames:
            if multidimensional:
                out[instrument_indexes[instrument], :] = GP_arguments[
                    instrument]
            else:
                out[instrument_indexes[instrument]] = GP_arguments[instrument]
        return out

    def sort_GP(self, dictype):
        if dictype == 'lc':
            # Sort first times, fluxes, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_lc_arguments['lc'][:, 0])
            self.t_lc = self.t_lc[idx_sort]
            self.y_lc = self.y_lc[idx_sort]
            self.yerr_lc = self.yerr_lc[idx_sort]

            self.GP_lc_arguments['lc'][:,0] = self.GP_lc_arguments['lc'][idx_sort,0]

            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new
            # ordering:
            for instrument in self.inames_lc:
                new_instrument_indexes = np.zeros(
                    len(self.instrument_indexes_lc[instrument]))
                instrument_indexes = self.instrument_indexes_lc[instrument]
                counter = 0
                for i in instrument_indexes:
                    new_instrument_indexes[counter] = np.where(
                        i == idx_sort)[0][0]
                    counter += 1
                self.instrument_indexes_lc[
                    instrument] = new_instrument_indexes.astype('int')
        elif dictype == 'rv':
            # Sort first times, rvs, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_rv_arguments['rv'][:, 0])
            self.t_rv = self.t_rv[idx_sort]
            self.y_rv = self.y_rv[idx_sort]
            self.yerr_rv = self.yerr_rv[idx_sort]

            self.GP_rv_arguments['rv'][:,0] = self.GP_rv_arguments['rv'][idx_sort,0]

            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new
            # ordering:
            for instrument in self.inames_rv:
              
                new_instrument_indexes = np.zeros(
                    len(self.instrument_indexes_rv[instrument]))
                
                instrument_indexes = self.instrument_indexes_rv[instrument]
                
                counter = 0
                for i in instrument_indexes:

                    new_instrument_indexes[counter] = np.where(i == idx_sort)[0][0]
                    counter += 1
                    
                self.instrument_indexes_rv[instrument] = new_instrument_indexes.astype('int')

    def generate_datadict(self, dictype):
        """
        This generates the options dictionary for lightcurves, RVs, and everything else you want to fit. Useful for the
        fit, as it separaters options per instrument.

        :param dictype: (string)
            Defines the type of dictionary type. It can either be 'lc' (for the lightcurve dictionary) or 'rv' (for the
            radial-velocity one).
        """

        dictionary = {}

        if dictype == 'lc':

            inames = self.inames_lc
            ninstruments = self.ninstruments_lc
            instrument_supersamp = self.lc_instrument_supersamp
            n_supersamp = self.lc_n_supersamp
            exptime_supersamp = self.lc_exptime_supersamp
            numbering_planets = self.numbering_transiting_planets
            # Check if model is global based on the input prior names. If they include as instrument "rv", set to global model:
            self.global_lc_model = self.check_global('lc')
            global_model = self.global_lc_model
            #if global_model and (self.GP_lc_arguments is not None):
            #    self.GP_lc_arguments['lc'] = self.append_GP(len(self.t_lc), self.instrument_indexes_lc, self.GP_lc_arguments, inames)
            GP_regressors = self.GP_lc_arguments

        elif dictype == 'rv':

            inames = self.inames_rv
            ninstruments = self.ninstruments_rv
            instrument_supersamp = None
            n_supersamp = None
            exptime_supersamp = None
            numbering_planets = self.numbering_rv_planets
            # Check if model is global based on the input prior names. If they include as instrument "lc", set to global model:
            self.global_rv_model = self.check_global('rv')
            global_model = self.global_rv_model
            # If global_model is True, create an additional key in the GP_regressors array that will have all the GP regressors appended:
            #if global_model and (self.GP_rv_arguments is not None):
            #    self.GP_rv_arguments['rv'] = self.append_GP(len(self.t_rv), self.instrument_indexes_rv, self.GP_rv_arguments, inames)
            GP_regressors = self.GP_rv_arguments

        else:
            raise Exception(
                'INPUT ERROR: dictype not understood. Has to be either lc or rv.'
            )

        for i in range(ninstruments):

            instrument = inames[i]
            dictionary[instrument] = {}
            # Save if a given instrument will receive resampling (initialize this as False):
            dictionary[instrument]['resampling'] = False
            # Save if a given instrument has GP fitting ON (initialize this as False):
            dictionary[instrument]['GPDetrend'] = False
            # Save if transit fitting will be done for a given dataset/instrument (this is so users can fit photometry with, e.g., GPs):
            if dictype == 'lc':

                dictionary[instrument]['TransitFit'] = False
                dictionary[instrument]['TransitFitCatwoman'] = False
                dictionary[instrument]['EclipseFit'] = False
                dictionary[instrument]['PhaseCurveFit'] = False
                dictionary[instrument]['CowanAgolPCFit'] = False
                dictionary[instrument]['LambertPCFit'] = False
                dictionary[instrument]['KelpHomoPCFit'] = False
                dictionary[instrument]['KelpInhomoPCFit'] = False
                dictionary[instrument]['KelpThmPCFit'] = False
                dictionary[instrument]['TranEclFit'] = False

        if dictype == 'lc':

            # Extract limb-darkening law and parametrization to be used to explore limb-darkeining. If no limb-darkening law was given by the user, 
            # assume LD law depending on whether the user defined a prior for q1/u1 only for a given instrument (in which that instrument is set to 
            # the linear law) or a prior for q1/u1 and q2/u2, in which case we assume the user wants to use a quadratic law for that instrument. 
            # If user gave one limb-darkening law, assume that law for all instruments that have priors for q1/u1 and q2/u2 (if only q1/u1 is given, 
            # assume linear for those instruments). If LD laws given for every instrument, extract them:

            all_ld_laws = self.ld_laws.split(',')

            if len(all_ld_laws) == 1:

                for i in range(ninstruments):

                    instrument = inames[i]
                    coeff1_given = False
                    parametrization = 'kipping2013'
                    coeff2_given = False

                    for parameter in self.priors.keys():

                        if parameter[0:2] == 'q1' or parameter[0:2] == 'u1' or parameter[0:2] == 'c1':

                            if instrument in parameter.split('_')[1:]:

                                coeff1_given = True
                                
                                # Check which parametrization/law the user is choosing:
                                if parameter[0:2] == 'u1':

                                    parametrization = 'normal'

                                if parameter[0:2] == 'c1':

                                    parametrization = 'normal-nonlinear'

                        if parameter[0:2] == 'q2' or parameter[0:2] == 'u2':

                            if instrument in parameter.split('_')[1:]:

                                coeff2_given = True

                    if coeff1_given and (not coeff2_given):

                        if parametrization == 'normal-nonlinear':

                            dictionary[instrument]['ldlaw'] = 'nonlinear' 
                            dictionary[instrument]['ldparametrization'] = 'normal'

                        else:

                            dictionary[instrument]['ldlaw'] = 'linear'
                            dictionary[instrument]['ldparametrization'] = parametrization

                    elif coeff1_given and coeff2_given:

                        dictionary[instrument]['ldlaw'] = (
                            all_ld_laws[0].split('-')[-1]).split()[0].lower()

                        dictionary[instrument]['ldparametrization'] = parametrization

                    elif (not coeff1_given) and coeff2_given:
                      
                        raise Exception(
                            'INPUT ERROR: it appears q1/u1 for instrument ' +
                            instrument +
                            ' was not defined (but q2/u2 was) in the prior file.')
                        
                    elif (not coeff1_given) and (not coeff2_given):

                        dictionary[instrument]['ldlaw'] = 'none'
                        dictionary[instrument]['ldparametrization'] = 'none'

            else:

                # Extract limb-darkening law from user-input:
                for ld_law in all_ld_laws:

                    instrument, ld = ld_law.split('-')

                    dictionary[instrument.split()
                               [0]]['ldlaw'] = ld.split()[0].lower()

                # Now extract parametrization for each instrument depending on user priors file/dictionary:
                for i in range(ninstruments):

                    instrument = inames[i]
                    coeff1_given = False
                    parametrization = 'kipping2013'
                    coeff2_given = False

                    for parameter in self.priors.keys():

                        if parameter[0:2] == 'q1' or parameter[0:2] == 'u1' or parameter[0:2] == 'c1':

                            if instrument in parameter.split('_')[1:]:

                                coeff1_given = True 
     
                                # Check which parametrization the user is choosing:
                                if parameter[0:2] == 'u1':

                                    parametrization = 'normal'

                                if parameter[0:2] == 'c1':

                                    parametrization = 'normal-nonlinear'

                        if parameter[0:2] == 'q2' or parameter[0:2] == 'u2':

                            if instrument in parameter.split('_')[1:]:

                                coeff2_given = True 

                    if coeff1_given and (not coeff2_given):

                        if parametrization == 'normal-nonlinear':

                            dictionary[instrument]['ldparametrization'] = 'normal'

                        else:

                            dictionary[instrument]['ldparametrization'] = parametrization

                    elif coeff1_given and coeff2_given:

                        dictionary[instrument]['ldparametrization'] = parametrization

                    elif (not coeff1_given) and coeff2_given:
     
                        raise Exception(
                            'INPUT ERROR: it appears q1/u1 for instrument ' +
                            instrument +
                            ' was not defined (but q2/u2 was) in the prior file.')
     
                    elif (not coeff1_given) and (not coeff2_given):

                        dictionary[instrument]['ldlaw'] = 'none'
                        dictionary[instrument]['ldparametrization'] = 'none'

        # Extract supersampling parameters if given.
        # For now this only allows inputs from lightcurves; TODO: add supersampling for RVs.
        if instrument_supersamp is not None and dictype == 'lc':
            for i in range(len(instrument_supersamp)):
                if self.verbose:
                    print('\t Resampling detected for instrument ',
                          instrument_supersamp[i])
                dictionary[instrument_supersamp[i]]['resampling'] = True
                dictionary[
                    instrument_supersamp[i]]['nresampling'] = n_supersamp[i]
                dictionary[instrument_supersamp[i]][
                    'exptimeresampling'] = exptime_supersamp[i]

        # Check that user gave periods in chronological order. If not, raise an exception, tell the user and stop this madness.
        # Note we only check if fixed or normal/truncated normal. In the uniform or log-uniform cases, we trust the user knows
        # what they are doing. We don't touch the Beta case because that would be nuts to put in a prior anyways most of the time.
        cp_pnumber = np.array([])
        cp_period = np.array([])
        for pri in self.priors.keys():
            if pri[0:2] == 'P_':
                if self.priors[pri]['distribution'].lower() in [
                        'normal', 'truncated normal'
                ]:
                    cp_pnumber = np.append(cp_pnumber,
                                           int(pri.split('_')[-1][1:]))
                    cp_period = np.append(
                        cp_period, self.priors[pri]['hyperparameters'][0])
                elif self.priors[pri]['distribution'].lower() == 'fixed':
                    cp_pnumber = np.append(cp_pnumber,
                                           int(pri.split('_')[-1][1:]))
                    cp_period = np.append(cp_period,
                                          self.priors[pri]['hyperparameters'])
        if len(cp_period) > 1:
            idx = np.argsort(cp_pnumber)
            cP = cp_period[idx[0]]
            cP_idx = cp_pnumber[idx[0]]
            for cidx in idx[1:]:
                P = cp_period[cidx]
                if P > cP:
                    cP = P
                    cP_idx = cp_pnumber[cidx]
                else:
                    print('\n')
                    raise Exception('INPUT ERROR: planetary periods in the priors are not ordered in chronological order. '+\
                                    'Planet p{0:} has a period of {1:} days, while planet p{2:} has a period of {3:} days (P_p{0:}<P_p{2:}).'.format(int(cp_pnumber[cidx]),P,int(cP_idx),cP))

        # Now, if generating lightcurve dict, check whether for some photometric instruments only photometry, and not a
        # transit, will be fit. This is based on whether the user gave limb-darkening coefficients for a given photometric
        # instrument or not. If given, transit is fit. If not, no transit is fit. At the same time check if user wants to
        # fit TTVs for the desired instrument. For this latter, initialize as false for each instrument and only change to
        # true if the priors are found:
        if dictype == 'lc':
            for i in range(ninstruments):
                dictionary[inames[i]]['TTVs'] = {}
                for pi in numbering_planets:
                    dictionary[inames[i]]['TTVs'][pi] = {}
                    dictionary[inames[i]]['TTVs'][pi]['status'] = False
                    dictionary[inames[i]]['TTVs'][pi]['parametrization'] = 'dt'
                    dictionary[inames[i]]['TTVs'][pi]['transit_number'] = []

                for pri in self.priors.keys():

                    if pri[0:2] == 'q1' or pri[0:2] == 'u1' or pri[0:2] == 'c1':

                        if inames[i] in pri.split('_'):

                            dictionary[inames[i]]['TransitFit'] = True

                            if self.verbose:

                                print('\t Transit fit detected for instrument ',
                                      inames[i])

                    if pri[0:2] == 'p1':

                        # If CW defined on instrument, or, there's a single CW for all instruments:
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['TransitFit'] = True
                            dictionary[inames[i]]['TransitFitCatwoman'] = True

                            if self.verbose:
                            
                                print('\t Transit (catwoman) fit detected for instrument ', inames[i])

                    if pri[0:11] == 'phaseoffset':
                        # If phase-offset defined on instrument, or, there's a single one for all instruments:
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['PhaseCurveFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit detected for instrument ', inames[i])

                    if pri[0:2] == 'C1':
                        # To check if the given instrument has Cowan & Agol (2008) phase curve fitting
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['CowanAgolPCFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit (Cowan & Agol 2008) detected for instrument ', inames[i])

                    if pri[0:9] == 'aglambert':

                        # To check if the given instrument has Lambertian phase curve fitting
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['LambertPCFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit (Lambertian) detected for instrument ', inames[i])
                            
                            # If Lambertian phase curve model is detected, then we have to manually turn on occultation model
                            # Because we still need an occultation model, however, since there will not be any fp_p1 parameter
                            # the occultation model will not be turned on automatically. So, manually turning it on

                            dictionary[inames[i]]['EclipseFit'] = True

                    if pri[0:10] == 'singlescat':

                        # To check if the given instrument has Kelp homogeneous phase curve fitting
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['KelpHomoPCFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit (kelp homogeneous reflection) detected for instrument ', inames[i])
                            
                            # If Kelp homogeneous reflection phase curve model is detected, then we have to manually turn on occultation model
                            # Because we still need an occultation model, however, since there will not be any fp_p1 parameter
                            # the occultation model will not be turned on automatically. So, manually turning it on

                            dictionary[inames[i]]['EclipseFit'] = True

                    if pri[0:5] == 'cml11':

                        # To check if the given instrument has Kelp thermal emission phase curve fitting
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['KelpThmPCFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit (kelp thermal emission) detected for instrument ', inames[i])
                            
                            # If Kelp homogeneous thermal emission phase curve model is detected, then we have to manually turn on occultation model
                            # Because we still need an occultation model, however, since there will not be any fp_p1 parameter
                            # the occultation model will not be turned on automatically. So, manually turning it on

                            dictionary[inames[i]]['EclipseFit'] = True

                    if pri[0:6] == 'agkelp':

                        # To check if the given instrument has kelp inhomogeneous reflection phase curve fitting
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['KelpInhomoPCFit'] = True

                            if self.verbose:

                                print('\t Phase curve fit (kelp inhomogeneous reflection) detected for instrument ', inames[i])

                            # If kelp inhomogeneous reflection phase curve model is detected, then we have to manually turn on occultation model
                            # Because we still need an occultation mode, however, since there will not be any fp_p1 parameter
                            # the occultation model will not be turned on automatically. So, manually turning it on

                            dictionary[inames[i]]['EclipseFit'] = True


                    if pri[0:2] == 'fp':

                        # If an eclipse for instrument or there's a single depth for all instruments:
                        if (inames[i] in pri.split('_')) or (len(pri.split('_')) == 2):

                            dictionary[inames[i]]['EclipseFit'] = True

                            if self.verbose:
                                print('\t Eclipse fit detected for instrument ',inames[i])

                    if pri[0:2] == 'dt' or pri[0:2] == 'T_':
                      
                        planet_number, instrument, ntransit = pri.split('_')[1:]

                        if inames[i] == instrument:
                            if pri[0:2] == 'T_':
                                dictionary[inames[i]]['TTVs'][int(
                                    planet_number[1:])]['parametrization'] = 'T'

                            dictionary[inames[i]]['TTVs'][int(
                                planet_number[1:])]['status'] = True
                            dictionary[inames[i]]['TTVs'][int(
                                planet_number[1:])]['transit_number'].append(
                                    int(ntransit))

                if dictionary[inames[i]]['TransitFit'] and dictionary[inames[i]]['EclipseFit']:

                    dictionary[inames[i]]['TranEclFit'] = True
                    dictionary[inames[i]]['TransitFit'] = False
                    dictionary[inames[i]]['EclipseFit'] = False

                    if self.verbose:

                        print('\t Joint Transit and Eclipse fit detected for instrument ',inames[i])

            for pi in numbering_planets:
                for i in range(ninstruments):
                    if dictionary[inames[i]]['TTVs'][pi]['status']:

                        dictionary[
                            inames[i]]['TTVs'][pi]['totalTTVtransits'] = len(
                                dictionary[
                                    inames[i]]['TTVs'][pi]['transit_number'])

        # Now, implement noise models for each of the instrument. First check if model should be global or instrument-by-instrument,
        # based on the input instruments given for the GP regressors.
        if global_model:
            dictionary['global_model'] = {}
            if GP_regressors is not None:
                dictionary['global_model']['GPDetrend'] = True
                dictionary['global_model']['noise_model'] = gaussian_process(
                    self,
                    model_type=dictype,
                    instrument=dictype,
                    matern_eps=self.matern_eps, 
                    george_hodlr=self.george_hodlr)
                if not dictionary['global_model']['noise_model'].isInit:
                    # If not initiated, most likely kernel is a celerite one. Reorder times, values, etc. This is OK --- is expected:
                    if dictype == 'lc':
                        self.sort_GP('lc')
                    elif dictype == 'rv':
                        self.sort_GP('rv')
                    # Try again:
                    dictionary['global_model'][
                        'noise_model'] = gaussian_process(
                            self,
                            model_type=dictype,
                            instrument=dictype,
                            matern_eps=self.matern_eps,
                            george_hodlr=self.george_hodlr)
                    if not dictionary['global_model']['noise_model'].isInit:
                        # Check, blame the user:
                        raise Exception(
                            'INPUT ERROR: GP initialization for object for ' +
                            dictype + ' global kernel failed.')
            else:
                dictionary['global_model']['GPDetrend'] = False
        else:
            for i in range(ninstruments):
                instrument = inames[i]

                if (GP_regressors is not None) and (instrument in GP_regressors.keys()):

                    dictionary[instrument]['GPDetrend'] = True
                    dictionary[instrument]['noise_model'] = gaussian_process(
                        self,
                        model_type=dictype,
                        instrument=instrument,
                        matern_eps=self.matern_eps,
                        george_hodlr=self.george_hodlr)
                    if not dictionary[instrument]['noise_model'].isInit:
                        # Blame the user, although perhaps we could simply solve this as for the global modelling?:
                        raise Exception(
                            'INPUT ERROR: GP regressors for instrument ' +
                            instrument +
                            ' use celerite, and are not in ascending or descending order. Please, give the input in those orders --- it will not work othersie.'
                        )

        # Check which eccentricity parametrization is going to be used for each planet in the juliet numbering scheme.
        # 0 = ecc, omega  1: ecosomega,esinomega  2: sqrt(e)cosomega, sqrt(e)sinomega
        dictionary['ecc_parametrization'] = {}
        if dictype == 'lc':
            dictionary['efficient_bp'] = {}
        for i in numbering_planets:
            if 'ecosomega_p' + str(i) in self.priors.keys():
                dictionary['ecc_parametrization'][i] = 1
                if self.verbose:
                    print(
                        '\t >> ecosomega,esinomega parametrization detected for '
                        + dictype + ' planet p' + str(i))
            elif 'secosomega_p' + str(i) in self.priors.keys():
                dictionary['ecc_parametrization'][i] = 2
                if self.verbose:
                    print(
                        '\t >> sqrt(e)cosomega, sqrt(e)sinomega parametrization detected for '
                        + dictype + ' planet p' + str(i))
            else:
                dictionary['ecc_parametrization'][i] = 0
                if self.verbose:
                    print('\t >> ecc,omega parametrization detected for ' +
                          dictype + ' planet p' + str(i))
            if dictype == 'lc':
                # Check if Espinoza (2018), (b,p) parametrization is on:
                if 'r1_p' + str(i) in self.priors.keys():
                    dictionary['efficient_bp'][i] = True
                    if self.verbose:
                        print('\t >> (b,p) parametrization detected for ' +
                              dictype + ' planet p' + str(i))
                else:
                    dictionary['efficient_bp'][i] = False

        # Check if stellar density is in the prior:
        if dictype == 'lc':
            dictionary['fitrho'] = False
            if 'rho' in self.priors.keys():
                dictionary['fitrho'] = True

        # For RV dictionaries, check if RV trend will be fitted:
        if dictype == 'rv':
            dictionary['fitrvline'] = False
            dictionary['fitrvquad'] = False
            if 'rv_slope' in self.priors.keys():
                if 'rv_quad' in self.priors.keys():
                    dictionary['fitrvquad'] = True
                    if self.verbose:
                        print('\t Fitting quadratic trend to RVs.')
                else:
                    dictionary['fitrvline'] = True
                    if self.verbose:
                        print('\t Fitting linear trend to RVs.')

        # Save dictionary to self:
        if dictype == 'lc':
            self.lc_options = dictionary
        elif dictype == 'rv':
            self.rv_options = dictionary
        else:
            raise Exception(
                'INPUT ERROR: dictype not understood. Has to be either lc or rv.'
            )

    def set_lc_data(self, t_lc, y_lc, yerr_lc, instruments_lc,
                    instrument_indexes_lc, ninstruments_lc, inames_lc,
                    lm_lc_boolean, lm_lc_arguments, nlm_lc_boolean):
        self.t_lc = t_lc.astype('float64')
        self.y_lc = y_lc
        self.yerr_lc = yerr_lc
        self.inames_lc = inames_lc
        self.instruments_lc = instruments_lc
        self.ninstruments_lc = ninstruments_lc
        self.instrument_indexes_lc = instrument_indexes_lc
        self.lm_lc_boolean = lm_lc_boolean
        self.lm_lc_arguments = lm_lc_arguments
        self.nlm_lc_boolean = nlm_lc_boolean
        self.lc_data = True

    def set_rv_data(self, t_rv, y_rv, yerr_rv, instruments_rv,
                    instrument_indexes_rv, ninstruments_rv, inames_rv,
                    lm_rv_boolean, lm_rv_arguments, nlm_rv_boolean):
        self.t_rv = t_rv.astype('float64')
        self.y_rv = y_rv
        self.yerr_rv = yerr_rv
        self.inames_rv = inames_rv
        self.instruments_rv = instruments_rv
        self.ninstruments_rv = ninstruments_rv
        self.instrument_indexes_rv = instrument_indexes_rv
        self.lm_rv_boolean = lm_rv_boolean
        self.lm_rv_arguments = lm_rv_arguments
        self.nlm_rv_boolean = nlm_rv_boolean
        self.rv_data = True

    def save(self):

        if self.out_folder[-1] != '/':
            self.out_folder = self.out_folder + '/'

        # First, save lightcurve data:
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder, exist_ok=True)
        if (not os.path.exists(self.out_folder + 'lc.dat')):
            if self.lcfilename is not None:
                os.system('cp ' + self.lcfilename + ' ' + self.out_folder +
                          'lc.dat')
            elif self.t_lc is not None:
                self.save_data(self.out_folder + 'lc.dat', self.t_lc, self.y_lc,
                               self.yerr_lc, self.instruments_lc,
                               self.lm_lc_boolean, self.lm_lc_arguments)

        # Now radial-velocity data:
        if (not os.path.exists(self.out_folder + 'rvs.dat')):
            if self.rvfilename is not None:
                os.system('cp ' + self.rvfilename + ' ' + self.out_folder +
                          'rvs.dat')
            elif self.t_rv is not None:
                self.save_data(self.out_folder + 'rvs.dat', self.t_rv,
                               self.y_rv, self.yerr_rv, self.instruments_rv,
                               self.lm_rv_boolean, self.lm_rv_arguments)

        # Next, save GP regressors:
        if (not os.path.exists(self.out_folder + 'GP_lc_regressors.dat')):
            if self.GPlceparamfile is not None:
                os.system('cp ' + self.GPlceparamfile + ' ' + self.out_folder +
                          'GP_lc_regressors.dat')
            elif self.GP_lc_arguments is not None:
                self.save_regressors(self.out_folder + 'GP_lc_regressors.dat',
                                     self.GP_lc_arguments)
        if (not os.path.exists(self.out_folder + 'GP_rv_regressors.dat')):
            if self.GPrveparamfile is not None:
                os.system('cp ' + self.GPrveparamfile + ' ' + self.out_folder +
                          'GP_rv_regressors.dat')
            elif self.GP_rv_arguments is not None:
                self.save_regressors(self.out_folder + 'GP_rv_regressors.dat',
                                     self.GP_rv_arguments)

        # Finally, save LM regressors if any:
        if (not os.path.exists(self.out_folder + 'LM_lc_regressors.dat')):
            if self.LMlceparamfile is not None:
                os.system('cp ' + self.LMlceparamfile + ' ' + self.out_folder +
                          'LM_lc_regressors.dat')
            elif self.LM_lc_arguments is not None:
                self.save_regressors(self.out_folder + 'LM_lc_regressors.dat',
                                     self.LM_lc_arguments)
        if (not os.path.exists(self.out_folder + 'LM_rv_regressors.dat')):
            if self.LMrveparamfile is not None:
                os.system('cp ' + self.LMrveparamfile + ' ' + self.out_folder +
                          'LM_rv_regressors.dat')
            elif self.LM_rv_arguments is not None:
                self.save_regressors(self.out_folder + 'LM_rv_regressors.dat',
                                     self.LM_rv_arguments)

        # Save priors:
        if (not os.path.exists(self.out_folder + 'priors.dat')):
            self.prior_fname = self.out_folder + 'priors.dat'
            self.save_priorfile(self.out_folder + 'priors.dat')

    def fit(self, **kwargs):
        """
        Perhaps the most important function of the juliet data object. This function fits your data using the nested
        sampler of choice. This returns a results object which contains all the posteriors information.
        """
        # Note this return call creates a fit *object* with the current data object. The fit class definition is below.
        return fit(self, **kwargs)

    def __init__(self,priors = None, starting_point = None, input_folder = None, t_lc = None, y_lc = None, yerr_lc = None, \
                 t_rv = None, y_rv = None, yerr_rv = None, GP_regressors_lc = None, linear_regressors_lc = None, \
                 GP_regressors_rv = None, linear_regressors_rv = None,
                 out_folder = None, lcfilename = None, rvfilename = None, GPlceparamfile = None,\
                 GPrveparamfile = None, LMlceparamfile = None, LMrveparamfile = None, lctimedef = 'TDB', rvtimedef = 'UTC',\
                 ld_laws = 'quadratic', priorfile = None, lc_n_supersamp = None, lc_exptime_supersamp = None, \
                 lc_instrument_supersamp = None, mag_to_flux = True, verbose = False, matern_eps = 0.01, george_hodlr = True, \
                 pickle_encoding = None, non_linear_functions = {}, extra_loglikelihood = None, backend = 'jax',
                 legacy_gp_parametrization = None):

        self.backend = 'jax'
        self.legacy_gp_parametrization = False if legacy_gp_parametrization is None else legacy_gp_parametrization
        self.lcfilename = lcfilename
        self.rvfilename = rvfilename
        self.GPlceparamfile = GPlceparamfile
        self.GPrveparamfile = GPrveparamfile
        self.LMlceparamfile = LMlceparamfile
        self.LMrveparamfile = LMrveparamfile
        self.verbose = verbose
        self.pickle_encoding = pickle_encoding
        self.starting_point = starting_point

        # GP options:
        self.matern_eps = matern_eps  # Epsilon parameter for celerite Matern32Term
        self.george_hodlr = george_hodlr # Wheter to use HODLR solver or not (see: http://dfm.io/george/current/user/solvers/)

        # Non-linear function options:
        self.non_linear_functions = non_linear_functions
        self.extra_loglikelihood = extra_loglikelihood

        if extra_loglikelihood is None:

            self.extra_loglikelihood_boolean = False

        else:

            self.extra_loglikelihood_boolean = True

        # Initialize data options for lightcurves:
        self.t_lc = None
        self.y_lc = None
        self.yerr_lc = None
        self.instruments_lc = None
        self.ninstruments_lc = None
        self.inames_lc = None
        self.instrument_indexes_lc = None
        self.lm_lc_boolean = None
        self.lm_lc_arguments = None
        self.GP_lc_arguments = None
        self.LM_lc_arguments = None
        self.lctimedef = lctimedef
        self.ld_laws = ld_laws
        self.lc_n_supersamp = lc_n_supersamp
        self.lc_exptime_supersamp = lc_exptime_supersamp
        self.lc_instrument_supersamp = lc_instrument_supersamp
        self.lc_data = False
        self.global_lc_model = False
        self.lc_options = {}

        # Initialize data options for RVs:
        self.t_rv = None
        self.y_rv = None
        self.yerr_rv = None
        self.instruments_rv = None
        self.ninstruments_rv = None
        self.inames_rv = None
        self.instrument_indexes_rv = None
        self.lm_rv_boolean = None
        self.lm_rv_arguments = None
        self.GP_rv_arguments = None
        self.LM_rv_arguments = None
        self.rvtimedef = rvtimedef
        self.rv_data = False
        self.global_rv_model = False
        self.rv_options = {}

        self.out_folder = None

        if input_folder is not None:
            if input_folder[-1] != '/':
                self.input_folder = input_folder + '/'
            else:
                self.input_folder = input_folder
            if os.path.exists(self.input_folder + 'lc.dat'):
                lcfilename = self.input_folder + 'lc.dat'
            if os.path.exists(self.input_folder + 'rvs.dat'):
                rvfilename = self.input_folder + 'rvs.dat'
            if (not os.path.exists(self.input_folder + 'lc.dat')) and (
                    not os.path.exists(self.input_folder + 'rvs.dat')):
                raise Exception('INPUT ERROR: No lightcurve data file (lc.dat) or radial-velocity data file (rvs.dat) found in folder '+self.input_folder+\
                                '. \n Create them and try again. For details, check juliet.load?')
            if os.path.exists(self.input_folder + 'GP_lc_regressors.dat'):
                GPlceparamfile = self.input_folder + 'GP_lc_regressors.dat'
            if os.path.exists(self.input_folder + 'GP_rv_regressors.dat'):
                GPrveparamfile = self.input_folder + 'GP_rv_regressors.dat'
            if os.path.exists(self.input_folder + 'LM_lc_regressors.dat'):
                LMlceparamfile = self.input_folder + 'LM_lc_regressors.dat'
            if os.path.exists(self.input_folder + 'LM_rv_regressors.dat'):
                LMrveparamfile = self.input_folder + 'LM_rv_regressors.dat'
            if os.path.exists(self.input_folder + 'priors.dat'):
                priors = self.input_folder + 'priors.dat'
            else:
                raise Exception('INPUT ERROR: Prior file (priors.dat) not found in folder '+self.input_folder+'.'+\
                                'Create it and try again. For details, check juliet.load?')
            # If there is an input folder and no out_folder, then simply set the out_folder as the input_folder
            # for ease in the later functions (more for replotting purposes)
            # So, one can simply do this to obtain the posteriors:
            # > dataset = juliet.load(input_folder=folder) # to reload the priors, data, etc.
            # > results = dataset.fit() # to obtain the results already found in the input_folder
            # > posteriors = results.posteriors
            if out_folder is None:
                self.out_folder = self.input_folder
        else:
            self.input_folder = None

        if type(priors) == str:

           self.prior_fname = priors
           priors, n_transit, n_rv, numbering_transit, numbering_rv, n_params, starting_point = readpriors(priors)
           # Save information stored in the prior: the dictionary, number of transiting planets,
           # number of RV planets, numbering of transiting and rv planets (e.g., if p1 and p3 transit
           # and all of them are RV planets, numbering_transit = [1,3] and numbering_rv = [1,2,3]).
           # Save also number of *free* parameters (FIXED don't count here).
           self.priors = priors
           self.n_transiting_planets = n_transit
           self.n_rv_planets = n_rv
           self.numbering_transiting_planets = numbering_transit
           self.numbering_rv_planets = numbering_rv
           self.nparams = n_params
           self.starting_point = starting_point
            
        elif type(priors) == dict:
            # Dictionary was passed, so save it (renaming sigma_w_rv_instrument to sigma_w_instrument, as done for prior files):
            priors = {('sigma_w_' + k.split('_')[-1] if k[:10] == 'sigma_w_rv' else k): v for k, v in priors.items()}
            self.priors = priors
            # Extract same info as above if-statement but using only the dictionary:
            n_transit, n_rv, numbering_transit, numbering_rv, n_params = readpriors(
                priors)
            # Save information:
            self.n_transiting_planets = n_transit
            self.n_rv_planets = n_rv
            self.numbering_transiting_planets = numbering_transit
            self.numbering_rv_planets = numbering_rv
            self.nparams = n_params
            self.prior_fname = None
        else:
            raise Exception(
                'INPUT ERROR: Prior file is not a string or a dictionary (and it has to). Do juliet.load? for details.'
            )

        # Define cases in which data is given through files:
        if (t_lc is None):
            if lcfilename is not None:
                t_lc,y_lc,yerr_lc,instruments_lc,instrument_indexes_lc,ninstruments_lc,inames_lc,lm_lc_boolean,lm_lc_arguments = \
                read_data(lcfilename)

                # Check if, for each instrument, linear regressors where given through arrays too:
                if linear_regressors_lc is not None:

                    for k in inames_lc:

                        if (k in linear_regressors_lc) and (lm_lc_boolean[k]):

                            # Raise exception if user gave both linear regressors both via arrays _and_ filename:
                            raise Exception(
                                'INPUT ERROR: linear regressors given both in '+lcfilename+' and via the linear_regressors_lc. Erase one; juliet cannot pick one over the other.'
                            )    

                        elif k in linear_regressors_lc:

                            lm_lc_boolean[k] = True 
                            lm_lc_arguments[k] = linear_regressors_lc[k]

                # Set null boolean for now for non-linear in data:
                nlm_lc_boolean = {} 
                for k in inames_lc:

                    nlm_lc_boolean[k] = False

                # Save data to object:

                self.set_lc_data(t_lc, y_lc, yerr_lc, instruments_lc,
                                 instrument_indexes_lc, ninstruments_lc,
                                 inames_lc, lm_lc_boolean, lm_lc_arguments, nlm_lc_boolean)

        if (t_rv is None):
            if rvfilename is not None:
                t_rv,y_rv,yerr_rv,instruments_rv,instrument_indexes_rv,ninstruments_rv,inames_rv,lm_rv_boolean,lm_rv_arguments = \
                read_data(rvfilename)

                # Check if, for each instrument, linear regressors where given through arrays too:
                if linear_regressors_rv is not None:

                    for k in inames_rv:

                        if (k in linear_regressors_rv) and (lm_rv_boolean[k]):

                            # Raise exception if user gave both linear regressors both via arrays _and_ filename:
                            raise Exception(
                                'INPUT ERROR: linear regressors given both in '+rvfilename+' and via the linear_regressors_rv. Erase one; juliet cannot pick one over the other.'
                            )

                        elif k in linear_regressors_rv:

                            lm_rv_boolean[k] = True
                            lm_rv_arguments[k] = linear_regressors_rv[k]

                # Set null boolean for now for non-linear in data:
                nlm_rv_boolean = {} 
                for k in inames_rv:

                    nlm_rv_boolean[k] = False

                # Save data to object:
                self.set_rv_data(t_rv, y_rv, yerr_rv, instruments_rv,
                                 instrument_indexes_rv, ninstruments_rv,
                                 inames_rv, lm_rv_boolean, lm_rv_arguments, nlm_rv_boolean)

        if (t_lc is None and t_rv is None):
            if (lcfilename is None) and (rvfilename is None):
                raise Exception('INPUT ERROR: No complete dataset (photometric or radial-velocity) given.\n'+\
                      ' Make sure to feed times (t_lc and/or t_rv), values (y_lc and/or y_rv), \n'+\
                      ' errors (yerr_lc and/or yerr_rv).')

        # Read GP regressors if given through files or arrays. The former takes priority. First lightcurve:
        if GPlceparamfile is not None:
            self.GP_lc_arguments, self.global_lc_model = readGPeparams(
                GPlceparamfile)
        elif GP_regressors_lc is not None:
            self.GP_lc_arguments = {k: np.array(v, copy=True) for k, v in GP_regressors_lc.items()}
            instruments = set(list(self.GP_lc_arguments.keys()))

        # Same thing for RVs:
        if GPrveparamfile is not None:
            self.GP_rv_arguments, self.global_rv_model = readGPeparams(
                GPrveparamfile)
        elif GP_regressors_rv is not None:
            self.GP_rv_arguments = {k: np.array(v, copy=True) for k, v in GP_regressors_rv.items()}
            instruments = set(list(self.GP_rv_arguments.keys()))

        # Same thing for linear regressors in case they were given in a separate file:
        if LMlceparamfile is not None:
            LM_lc_arguments, dummy_var = readGPeparams(LMlceparamfile)
            for lmi in list(LM_lc_arguments.keys()):
                lm_lc_boolean[lmi] = True
                lm_lc_arguments[lmi] = LM_lc_arguments[lmi]

        # Same thing for RVs:
        if LMrveparamfile is not None:
            LM_rv_arguments, dummy_var = readGPeparams(LMrveparamfile)
            for lmi in list(LM_rv_arguments.keys()):
                lm_rv_boolean[lmi] = True
                lm_rv_arguments[lmi] = LM_rv_arguments[lmi]

        # If data given through direct arrays (i.e., not data files), generate some useful internal lightcurve arrays: inames_lc, which have the different lightcurve instrument names,
        # instrument_indexes_lc (dictionary that holds, for each instrument, the indexes that have the time/lightcurve data for that particular instrument), lm_lc_boolean (dictionary of
        # booleans; True for an instrument if it has linear regressors), lm_lc_arguments (dictionary containing the linear regressors for each instrument), etc.:
        if (lcfilename is None) and (t_lc is not None):
            # First check user gave all data:
            input_error_catcher(t_lc, y_lc, yerr_lc, 'lightcurve')
            # Convert times to float64 (batman really hates non-float64 inputs):
            for instrument in t_lc.keys():
                t_lc[instrument] = t_lc[instrument].astype('float64')
            # Create global arrays:
            tglobal_lc, yglobal_lc, yglobalerr_lc, instruments_lc = self.convert_input_data(
                t_lc, y_lc, yerr_lc)
            # Save data in a format useful for global modelling:
            inames_lc, instrument_indexes_lc, lm_lc_boolean, nlm_lc_boolean = self.data_preparation(
                tglobal_lc, instruments_lc, linear_regressors_lc, non_linear_functions)
            lm_lc_arguments = linear_regressors_lc
            ninstruments_lc = len(inames_lc)

            # Save data to object:
            self.set_lc_data(tglobal_lc, yglobal_lc, yglobalerr_lc,
                             instruments_lc, instrument_indexes_lc,
                             ninstruments_lc, inames_lc, lm_lc_boolean,
                             lm_lc_arguments, nlm_lc_boolean)

            # Save input dictionaries:
            self.times_lc = t_lc
            self.data_lc = y_lc
            self.errors_lc = yerr_lc
        elif t_lc is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_lc, data_lc, errors_lc = self.convert_to_dictionary(
                t_lc, y_lc, yerr_lc, instrument_indexes_lc)
            self.times_lc = times_lc
            self.data_lc = data_lc
            self.errors_lc = errors_lc

        # Same for radial-velocity data:
        if (rvfilename is None) and (t_rv is not None):
            input_error_catcher(t_rv, y_rv, yerr_rv, 'radial-velocity')
            tglobal_rv, yglobal_rv, yglobalerr_rv, instruments_rv = self.convert_input_data(
                t_rv, y_rv, yerr_rv)
            inames_rv, instrument_indexes_rv, lm_rv_boolean, nlm_rv_boolean = self.data_preparation(
                tglobal_rv, instruments_rv, linear_regressors_rv, non_linear_functions)
            lm_rv_arguments = linear_regressors_rv
            ninstruments_rv = len(inames_rv)

            # Save data to object:

            self.set_rv_data(tglobal_rv, yglobal_rv, yglobalerr_rv,
                             instruments_rv, instrument_indexes_rv,
                             ninstruments_rv, inames_rv, lm_rv_boolean,
                             lm_rv_arguments, nlm_rv_boolean)

            # Save input dictionaries:
            self.times_rv = t_rv
            self.data_rv = y_rv
            self.errors_rv = yerr_rv
        elif t_rv is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_rv, data_rv, errors_rv = self.convert_to_dictionary(
                t_rv, y_rv, yerr_rv, instrument_indexes_rv)
            self.times_rv = times_rv
            self.data_rv = data_rv
            self.errors_rv = errors_rv

        # If out_folder does not exist, create it, and save data to it:
        if out_folder is not None:
            self.out_folder = out_folder
            self.save()

        # Finally, generate datadicts, that will save information about the fits, including gaussian_process objects for each instrument that requires it
        # (including the case of global models):
        if t_lc is not None:
            self.generate_datadict('lc')
        if t_rv is not None:
            self.generate_datadict('rv')



# Samplers available in the JAX backend. Nested sampling returns the log-evidence; the MCMC ones don't.
nested_samplers = ['nested', 'nautilus', 'ultranest', 'slicesampler_ultranest']
mcmc_samplers = ['nuts', 'emcee', 'zeus']
# Names of samplers of previous (non-JAX) juliet versions; these are now run with the JAX nested sampler:
legacy_nested_samplers = ['multinest', 'dynesty', 'dynamic_dynesty']


def _matching_kwargs(function, kwargs):
    """Extract the entries of kwargs that are arguments of function."""
    names = inspect.signature(function).parameters.keys()
    return {k: kwargs[k] for k in kwargs if k in names}


class fit(object):
    """
    Given a juliet data object, this class performs a fit to the data and returns a results object to explore the
    results. Example usage:

               >>> results = juliet.fit(data)

    :param data: (juliet object)
        An object containing all the information regarding the data to be fitted, including options of the fit.
        Generated via juliet.load().

    On top of ``data``, a series of extra keywords can be included:

    :param sampler: (optional, string)
        String defining the sampler to be used on the fit. All samplers are written in JAX:

        - ``nested`` (default): batched nested slice sampling (`blackjax <https://github.com/blackjax-devs/blackjax>`_'s ``nss``). Returns
          the log-evidence (``lnZ``) along with the posterior samples. On each iteration, ``num_delete`` live points are replaced in
          parallel. Names of the nested samplers of previous juliet versions (``multinest``, ``dynesty``, ``dynamic_dynesty``,
          ``ultranest``, ``slicesampler_ultranest``) are mapped to this sampler. It needs many likelihood evaluations (it is fast for
          cheap likelihoods, especially on GPUs), and for multimodal posteriors its ``lnZ`` can scatter between runs by much more than
          ``lnZerr`` (new points rarely move between modes); use ``nautilus`` (or compare several seeds) in that case.
        - ``nuts``: `numpyro <https://num.pyro.ai>`_'s No-U-Turn Sampler, with ``num_chains`` (default 4) chains run vectorized.
        - ``emcee``: numpyro's affine-invariant ensemble sampler (``AIES``; same algorithm as ``emcee``) with ``nwalkers`` walkers.
        - ``zeus``: numpyro's ensemble slice sampler (``ESS``; same algorithm as ``zeus``) with ``nwalkers`` walkers.
        - ``nautilus``: `nautilus <https://nautilus-sampler.readthedocs.io>`_'s neural-network-boosted importance nested
          sampling (requires the ``nautilus-sampler`` package), with the JAX likelihood evaluated in batches. Returns ``lnZ``;
          ``lnZerr`` is the approximate importance-sampling error ``1/sqrt(N_eff)``.

    :param n_live_points: (optional, int)
        Number of live-points to use on the nested sampler. Default is 500.

    :param nwalkers: (optional if using emcee or zeus, int)
        Number of walkers to use by the ensemble samplers. Default is 100.

    :param nsteps: (optional if using MCMC, int)
        Number of steps/jumps (per chain/walker) to perform on the MCMC run after the burn-in. Default is 300.

    :param nburnin: (optional if using MCMC, int)
        Number of burnin (warm-up) steps/jumps when performing the MCMC run. Default is 500.

    :param emcee_factor: (optional, for emcee and zeus only, float)
        Factor multiplying the standard-gaussian ball around which the initial position is perturbed for each walker. Default is 1e-4.

    :param ecclim: (optional, float)
        Upper limit on the maximum eccentricity to sample. Default is ``1``.

    :param pl: (optional, float)
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled.
        Default is ``0``.

    :param pu: (optional, float)
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ta: (optional, float)
        Time to be substracted to the input times in order to generate the linear and/or quadratic trend to be added to the model.
        Default is 2458460.

    :param light_travel_delay: (optinal, bool)
        Boolean indicating if light travel time delay wants to be included on eclipse time calculations.

    :param stellar_radius: (optional, float)
        Stellar radius in units of solar-radii to use for the light travel time corrections.

    :param kelp_refl_interpolation_knots: (optional, int)
        Number of knots in case interpolation is to be performed along phases for kelp reflection phase curves. Default is None.

    :param seed: (optional, int)
        Seed for the random number generator of the samplers. Default is a random seed.

    Extra keywords are passed to the sampler of choice:

    - ``nested``: ``num_delete`` (number of live points replaced in parallel per iteration; default ``n_live_points // 2``),
      ``num_inner_steps`` (slice-sampling steps per new live point; default ``max(5, 2 * ndim)``), ``dlogz`` (stopping criterion on
      the remaining evidence; default 0.1), ``n_posterior_samples`` (number of equally-weighted posterior samples to return;
      default ``max(ESS, 1000)``) and ``max_iterations``.
    - ``nuts``: ``num_chains`` (default 4) and any argument of ``numpyro.infer.NUTS`` (e.g., ``target_accept_prob``). By default, a dense
      mass matrix is adapted without regularization (``dense_mass = True``, ``regularize_mass_matrix = False``).
    - ``emcee``/``zeus``: any argument of ``numpyro.infer.AIES``/``numpyro.infer.ESS`` (e.g., ``moves``).
    - All MCMCs: any argument of ``numpyro.infer.MCMC`` (e.g., ``thinning``, ``progress_bar``).
    - ``nautilus``: ``n_live_points`` is nautilus' ``n_live`` (nautilus recommends 1000--3000); any argument of
      ``nautilus.Sampler`` (e.g., ``n_batch``, ``n_networks``) or of ``nautilus.Sampler.run`` (e.g., ``f_live``, ``n_eff``).
      nautilus spends much of its run time training neural networks on one CPU core. To train them in parallel while the
      batched likelihood stays in the main process, pass a pool for its sampler calculations, e.g.,
      ``pool = (None, multiprocessing.get_context('fork').Pool(4))`` (a 'fork' pool does not re-import the calling script).
    - ``ultranest`` / ``slicesampler_ultranest`` (needs ``ultranest``): UltraNest's ``ReactiveNestedSampler`` with a vectorized
      likelihood; ``n_live_points`` is ``min_num_live_points``. ``ultranest`` samples from UltraNest's MLFriends regions
      (best in low dimensions); ``slicesampler_ultranest`` uses its vectorized ``PopulationSliceSampler``, with ``popsize`` walkers
      evolved at once (default ``n_live_points // 2``), ``num_inner_steps`` slice steps per new point (default
      ``max(5, 2 * ndim)``) and ``generate_direction`` (default ``generate_mixture_random_direction``). Any argument of
      ``ultranest.ReactiveNestedSampler`` (e.g., ``ndraw_max``, ``log_dir`` to write UltraNest's output files) or of its ``run``
      method (e.g., ``dlogz``, ``frac_remain``, ``max_num_improvement_loops``) can be given.

    Note that, as the likelihood is jit-compiled, any ``non_linear_functions`` or ``extra_loglikelihood`` given to ``juliet.load``
    are best written with ``jax.numpy``. Functions written with NumPy are evaluated through ``jax.pure_callback`` (slower, and not
    available with ``sampler = 'nuts'``, which needs gradients).

    The following keywords from previous juliet versions are accepted but have no effect in the JAX backend: ``nthreads``,
    ``use_ultranest``, ``use_dynesty``, ``dynamic``, ``dynesty_bound``, ``dynesty_sample``, ``dynesty_nthreads``,
    ``dynesty_n_effective``, ``dynesty_use_stop``, ``dynesty_use_pool``, ``dynesty_save_states`` and ``dynesty_resume``.
    """

    def __new__(cls, data, *args, **kwargs):
        # Data loaded with backend = 'legacy' are handled by the legacy implementation:
        if getattr(data, 'backend', 'jax') == 'legacy':
            from .legacy.fit import fit as legacy_class
            return legacy_class(data, *args, **kwargs)
        return super().__new__(cls)

    # Map between the free parameter vector x (physical units) and the parameter dictionary used by the models:
    def parameter_dictionary(self, x):
        pv = dict(self.fixed_values)
        for i, pname in enumerate(self.paramnames):
            pv[pname] = x[i]
        return pv

    def _loglike_x(self, x):
        pv = self.parameter_dictionary(x)
        log_likelihood = 0.
        if self.data.t_lc is not None:
            log_likelihood = log_likelihood + self.lc.log_likelihood_fn(pv)
        if self.data.t_rv is not None:
            log_likelihood = log_likelihood + self.rv.log_likelihood_fn(pv)
        # Evaluate any extra likelihoods:
        if self.extra_loglikelihood_boolean:
            log_likelihood = log_likelihood + self._extra_loglikelihood_fn(pv)
        return jnp.where(jnp.isnan(log_likelihood), -jnp.inf, log_likelihood)

    def _logprior_x(self, x):
        return sum(d.log_prob(x[i]) for i, d in enumerate(self.prior_dists))

    # The samplers work on an unconstrained vector z; x = T(z) maps it to the support of each prior:
    def _x_of_z(self, z):
        return jnp.stack([T(z[i]) for i, T in enumerate(self.transforms)])

    def _z_of_x(self, x):
        z = jnp.stack([T.inv(x[..., i]) for i, T in enumerate(self.transforms)], axis=-1)
        return jnp.where(jnp.isfinite(z), z, 0.)

    def _logprior_z(self, z):
        lp = 0.
        for i, (d, T) in enumerate(zip(self.prior_dists, self.transforms)):
            xi = T(z[i])
            lp = lp + d.log_prob(xi) + T.log_abs_det_jacobian(z[i], xi)
        return lp

    def _loglike_z(self, z):
        return self._loglike_x(self._x_of_z(z))

    def _potential_z(self, z):
        lp = self._logprior_z(z) + self._loglike_z(z)
        return jnp.where(jnp.isnan(lp), jnp.inf, -lp)

    # Methods kept for back-compatibility (these take numpy arrays with the free parameters, in physical units):
    def loglike(self, theta, ndim=None, nparams=None):
        return float(self._loglike_x_jit(jnp.asarray(theta, dtype=float)))

    def logprior(self, theta):
        return float(self._logprior_x_jit(jnp.asarray(theta, dtype=float)))

    def logprob(self, theta):
        lp = self.logprior(theta) + self.loglike(theta)
        return -np.inf if np.isnan(lp) else lp

    def _sample_prior(self, key, n):
        keys = jax.random.split(key, len(self.prior_dists))
        return jnp.stack([d.sample(keys[i], (n,)) for i, d in enumerate(self.prior_dists)], axis=-1)

    def _best_prior_draws(self, key, n):
        # Initial positions for MCMCs if no starting point is given: the n prior draws with the highest posterior
        # probability out of max(1000, 20 * n):
        x = self._sample_prior(key, max(1000, 20 * n))
        z = self._z_of_x(x)
        lp = -jax.lax.map(jax.jit(self._potential_z), z, batch_size=self.batch_size)
        return z[jnp.argsort(-lp)[:n]]

    def _unit_cube_prior(self):
        # Vectorized prior transform of the unit hypercube for samplers that work there (nautilus, ultranest), with the
        # same transforms as juliet's original nested samplers:
        transforms = {'uniform': transform_uniform, 'normal': transform_normal, 'truncatednormal': transform_truncated_normal,
                      'jeffreys': transform_loguniform, 'loguniform': transform_loguniform, 'beta': transform_beta,
                      'exponential': transform_exponential, 'modjeffreys': transform_modifiedjeffreys}
        prior_transforms = [(transforms[self.data.priors[p]['distribution'].lower()], self.data.priors[p]['hyperparameters'])
                            for p in self.paramnames]

        def prior(u):
            x = np.empty_like(u)
            for i, (transform, hyperparameters) in enumerate(prior_transforms):
                x[:, i] = transform(u[:, i], hyperparameters)
            return x

        return prior

    def _host_loglike(self, pad=False):
        # Log-likelihood of an (n, nparams) NumPy array of physical parameters, evaluated in one batch, for samplers that
        # run on the host (nautilus, ultranest). With pad = True, batches are padded to powers of two so that samplers
        # proposing varying numbers of points don't trigger a new compilation for every batch size.
        batched_loglike = jax.jit(jax.vmap(chunked_vmap(self._loglike_x_jit, self.batch_size)))

        def likelihood(x):
            n = len(x)
            if n == 0:
                return np.zeros(0)
            if pad:
                n_padded = max(64, 1 << (n - 1).bit_length())
                x = np.concatenate([x, np.repeat(x[:1], n_padded - n, axis=0)])
            log_like = np.asarray(batched_loglike(jnp.asarray(x)))[:n]
            return np.where(np.isfinite(log_like), log_like, -1e300)

        return likelihood

    def __init__(self, data, sampler = 'nested', n_live_points = 500, nwalkers = 100, nsteps = 300, nburnin = 500, emcee_factor = 1e-4, \
                 ecclim = 1., pl = 0.0, pu = 1.0, ta = 2458460., nthreads = None, light_travel_delay = False, stellar_radius = None, \
                 kelp_refl_interpolation_knots = None, kelp_thm_interpolation_knots = None, kelp_filt_wav = None, kelp_filt_trans = None,\
                 stellar_teff = None, kelp_ntheta = 5, kelp_nphi = 75, use_ultranest = False, use_dynesty = False, dynamic = False,\
                 dynesty_bound = 'multi', dynesty_sample='rwalk', dynesty_nthreads = None, dynesty_n_effective = np.inf, dynesty_use_stop = True,\
                 dynesty_use_pool = None, dynesty_save_states=False, dynesty_resume=False, seed = None, **kwargs):

        # Define output results object:
        self.results = None

        # Now extract sampler options:
        self.sampler = sampler
        self.n_live_points = n_live_points
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.emcee_factor = emcee_factor
        self.nburnin = nburnin
        self.nthreads = nthreads
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)

        # Kelp options (see juliet.model):
        self.kelp_refl_interpolation_knots = kelp_refl_interpolation_knots
        self.kelp_thm_interpolation_knots = kelp_thm_interpolation_knots
        self.kelp_filt_wav = kelp_filt_wav
        self.kelp_filt_trans = kelp_filt_trans
        self.stellar_teff = stellar_teff
        self.kelp_ntheta = kelp_ntheta
        self.kelp_nphi = kelp_nphi

        # Extract physical model details:
        self.light_travel_delay = light_travel_delay
        if self.light_travel_delay and (stellar_radius is None):

            raise Exception('Error: if light_travel_delay is activated, a stellar radius needs to be given as well via stellar_radius = yourvalue; e.g., dataset.fit(..., light_travel_delay = True, stellar_radius = 1.1234).')

        self.stellar_radius = stellar_radius

        # Deprecated sampler flags:
        if use_ultranest:
            print('WARNING: use_ultranest is deprecated; use sampler = "ultranest".')
            self.sampler = 'ultranest'
        elif use_dynesty:
            print('WARNING: use_dynesty is deprecated; using the JAX nested sampler (sampler = "nested").')
            self.sampler = 'nested'
        if self.sampler in legacy_nested_samplers:
            print('Note: juliet now runs on JAX; sampler "' + self.sampler + '" is replaced by the JAX nested sampler (sampler = "nested").')
            self.sampler = 'nested'
        if 'ultranest' in self.sampler:
            try:
                import ultranest
            except ImportError:
                print('Note: ultranest is not installed; using the JAX nested sampler (sampler = "nested") instead.')
                self.sampler = 'nested'
        if self.sampler not in nested_samplers + mcmc_samplers:
            raise Exception('INPUT ERROR: sampler "' + self.sampler + '" not recognized. Options are: ' + ', '.join(nested_samplers + mcmc_samplers) + '.')
        if self.nthreads is not None:
            print('Note: nthreads has no effect in the JAX backend (likelihood evaluations are vectorized instead).')

        # Define (exo-)algorithmic options:
        self.ecclim = ecclim
        self.pl = pl
        self.pu = pu
        self.ta = ta

        # Inhert data object:
        self.data = data

        # Inhert the output folder:
        self.out_folder = data.out_folder

        # Inhert extra likelihood:
        self.extra_loglikelihood = data.extra_loglikelihood
        self.extra_loglikelihood_boolean = data.extra_loglikelihood_boolean

        # Prefix of the output files:
        self.sampler_prefix = self.sampler + '_'

        # Generate a posteriors self that will save the current values of each of the parameters. Initialization value is unimportant for nested samplers;
        # if MCMC, this saves the initial parameter values:
        self.posteriors = {}
        self.model_parameters = list(self.data.priors.keys())
        self.paramnames = []
        self.fixed_values = {}
        self.prior_dists = []
        for pname in self.model_parameters:

            if self.data.priors[pname]['distribution'].lower() == 'fixed':
                self.posteriors[pname] = self.data.priors[pname]['hyperparameters']
                self.fixed_values[pname] = float(self.data.priors[pname]['hyperparameters'])

            else:
                if self.sampler in mcmc_samplers and self.data.starting_point is not None:
                    self.posteriors[pname] = self.data.starting_point[pname]
                else:
                    self.posteriors[pname] = 0.
                self.paramnames.append(pname)
                self.prior_dists.append(jm.prior_distribution(self.data.priors[pname]['distribution'],
                                                              self.data.priors[pname]['hyperparameters']))

        self.transforms = [biject_to(d.support) for d in self.prior_dists]
        self.nparams = len(self.paramnames)

        # Generate light-curve and radial-velocity models:
        if self.data.t_lc is not None:
            self.lc = model(self.data,
                            modeltype='lc',
                            pl=self.pl,
                            pu=self.pu,
                            ecclim=self.ecclim,
                            light_travel_delay = self.light_travel_delay,
                            stellar_radius = self.stellar_radius,
                            kelp_refl_interpolation_knots=self.kelp_refl_interpolation_knots,
                            kelp_thm_interpolation_knots=self.kelp_thm_interpolation_knots,
                            kelp_filt_wav=self.kelp_filt_wav,
                            kelp_filt_trans=self.kelp_filt_trans,
                            stellar_teff=self.stellar_teff,
                            kelp_ntheta=self.kelp_ntheta,
                            kelp_nphi=self.kelp_nphi,
                            log_like_calc=True)
        if self.data.t_rv is not None:

            self.rv = model(self.data,
                            modeltype='rv',
                            ecclim=self.ecclim,
                            ta=self.ta,
                            log_like_calc=True)

        # Number of likelihood evaluations vectorized at once when samplers evaluate many points (bounds the memory used
        # for large datasets; see model.batch_size):
        self.batch_size = kwargs.get('batch_size', min(m.batch_size for m in [getattr(self, 'lc', None), getattr(self, 'rv', None)]
                                                       if m is not None))

        # Extra log-likelihood (functions not written with jax.numpy are evaluated on the host via callbacks):
        uses_callbacks = any(m.uses_callbacks for m in [getattr(self, 'lc', None), getattr(self, 'rv', None)] if m is not None)
        if self.extra_loglikelihood_boolean:
            self._extra_loglikelihood_fn, is_callback = jm.jax_compatible(self.extra_loglikelihood['loglikelihood'],
                                                                          jm.example_parameter_values(self.data.priors),
                                                                          'extra_loglikelihood')
            uses_callbacks = uses_callbacks or is_callback
        if uses_callbacks and self.sampler == 'nuts':
            raise Exception('sampler = "nuts" needs gradients of the likelihood, so non_linear_functions and extra_loglikelihood '
                            'must be written with jax.numpy (or use the nested, emcee or zeus samplers).')

        # jit-compiled log-likelihood and log-prior of the free parameters (in physical units):
        self._loglike_x_jit = jax.jit(self._loglike_x)
        self._logprior_x_jit = jax.jit(self._logprior_x)

        # First, check if a run has already been performed with the user-defined sampler. If it hasn't, run it.
        # If it has (detected through its output filename), skip running again and jump straight to loading the
        # data:
        out = {}
        runSampler = False
        if self.out_folder is None:
            self.out_folder = os.getcwd() + '/'
        if (not os.path.exists(self.out_folder + self.sampler_prefix +
                               'posteriors.pkl')):
            runSampler = True

        # If runSampler is True, then run the sampler of choice:
        if runSampler:

            rng_key = jax.random.PRNGKey(self.seed)
            rng_key, init_key, run_key = jax.random.split(rng_key, 3)

            if self.sampler == 'nested':

                num_delete = kwargs.get('num_delete', max(1, self.n_live_points // 2))
                num_inner_steps = kwargs.get('num_inner_steps', max(5, 2 * self.nparams))
                if self.data.verbose:
                    print('\t Running nested sampling with {0:} live points, {1:} deleted per iteration and {2:} slice steps per new point.'.format(
                          self.n_live_points, num_delete, num_inner_steps))
                initial_z = self._z_of_x(self._sample_prior(init_key, self.n_live_points))

                ns = run_nested(jax.jit(self._logprior_z), chunked_vmap(jax.jit(self._loglike_z), self.batch_size), initial_z, run_key,
                                num_delete=num_delete, num_inner_steps=num_inner_steps,
                                dlogz=kwargs.get('dlogz', 0.1),
                                max_iterations=kwargs.get('max_iterations', 1000000),
                                n_posterior_samples=kwargs.get('n_posterior_samples', None),
                                verbose=self.data.verbose)

                x_of_z = jax.jit(jax.vmap(self._x_of_z))
                posterior_samples = np.asarray(x_of_z(ns['posterior_samples']))

                # Save nested sampling outputs (dead points, in physical units):
                out['nested_output'] = {'samples': np.asarray(x_of_z(ns['samples'])),
                                        'loglikelihood': ns['loglikelihood'],
                                        'logwt': ns['logwt'],
                                        'ess': ns['ess'],
                                        'niterations': ns['niterations']}
                out['lnZ'] = ns['logz']
                out['lnZerr'] = ns['logzerr']

            elif self.sampler == 'nautilus':

                import nautilus
                prior, likelihood = self._unit_cube_prior(), self._host_loglike()

                sampler_kwargs = dict(n_live=self.n_live_points, n_batch=kwargs.get('n_batch', 1000), seed=int(self.seed))
                sampler_kwargs.update(_matching_kwargs(nautilus.Sampler.__init__, kwargs))
                for k in ['prior', 'likelihood', 'n_dim', 'vectorized', 'pass_dict']:
                    sampler_kwargs.pop(k, None)
                run_kwargs = dict(verbose=self.data.verbose, discard_exploration=True)
                run_kwargs.update(_matching_kwargs(nautilus.Sampler.run, kwargs))
                ns = nautilus.Sampler(prior, likelihood, n_dim=self.nparams, vectorized=True, pass_dict=False, **sampler_kwargs)
                ns.run(**run_kwargs)
                points, log_w, log_l = ns.posterior()
                posterior_samples = np.asarray(ns.posterior(equal_weight=True)[0])
                out['nautilus_output'] = {'samples': points, 'logwt': log_w, 'loglikelihood': log_l, 'ess': ns.n_eff}
                out['lnZ'] = float(ns.log_z)
                out['lnZerr'] = float(1. / np.sqrt(ns.n_eff))

            elif 'ultranest' in self.sampler:

                import ultranest
                # UltraNest's region sampler proposes varying numbers of points per call, so batches are padded:
                prior, likelihood = self._unit_cube_prior(), self._host_loglike(pad=True)
                np.random.seed(self.seed % 2**32)  # (ultranest draws from NumPy's global random number generator)

                sampler_kwargs = dict(log_dir=None)
                sampler_kwargs.update(_matching_kwargs(ultranest.ReactiveNestedSampler.__init__, kwargs))
                for k in ['param_names', 'loglike', 'transform', 'vectorized']:
                    sampler_kwargs.pop(k, None)
                ns = ultranest.ReactiveNestedSampler(self.paramnames, likelihood, transform=prior, vectorized=True, **sampler_kwargs)

                if self.sampler == 'slicesampler_ultranest':
                    # Vectorized slice sampling: popsize walkers are evolved at once, each needing num_inner_steps slice steps
                    # (same default as the JAX nested sampler):
                    from ultranest import popstepsampler
                    ns.stepsampler = popstepsampler.PopulationSliceSampler(
                        popsize=kwargs.get('popsize', max(1, self.n_live_points // 2)),
                        nsteps=kwargs.get('num_inner_steps', max(5, 2 * self.nparams)),
                        generate_direction=kwargs.get('generate_direction', popstepsampler.generate_mixture_random_direction))

                # Same defaults as juliet's original ultranest samplers:
                run_kwargs = dict(min_num_live_points=self.n_live_points, frac_remain=0.1, max_num_improvement_loops=1,
                                  show_status=self.data.verbose, viz_callback=False)
                run_kwargs.update(_matching_kwargs(ultranest.ReactiveNestedSampler.run, kwargs))
                results = ns.run(**run_kwargs)

                posterior_samples = np.asarray(results['samples'])
                weighted = results['weighted_samples']
                out['ultranest_output'] = {'samples': np.asarray(weighted['points']), 'logwt': np.asarray(weighted['logw']),
                                           'loglikelihood': np.asarray(weighted['logl']), 'ess': float(results['ess']),
                                           'ncall': int(results['ncall'])}
                out['lnZ'] = float(results['logz'])
                out['lnZerr'] = float(results['logzerr'])

            else:

                from numpyro.infer import MCMC
                mcmc_kwargs = _matching_kwargs(MCMC.__init__, kwargs)
                for k in ['num_warmup', 'num_samples', 'num_chains', 'chain_method']:
                    mcmc_kwargs.pop(k, None)

                if self.sampler == 'nuts':

                    from numpyro.infer import NUTS
                    # Posterior widths of juliet parameters span many orders of magnitude (e.g., t0 vs. sigma_w), so by default
                    # adapt a dense mass matrix without numpyro's regularization (which shrinks variances towards 1e-3):
                    kernel_kwargs = {'dense_mass': True, 'regularize_mass_matrix': False}
                    kernel_kwargs.update(_matching_kwargs(NUTS.__init__, kwargs))
                    kernel_kwargs.pop('potential_fn', None)
                    nchains = kwargs.get('num_chains', 4)
                    kernel_name = 'nuts'

                else:

                    from numpyro.infer import AIES, ESS
                    kernel_class = AIES if self.sampler == 'emcee' else ESS
                    # Don't shuffle walkers between iterations (numpyro's default for ESS), so posteriors_per_walker
                    # holds the actual trajectory of each walker:
                    kernel_kwargs = {'randomize_split': False}
                    kernel_kwargs.update(_matching_kwargs(kernel_class.__init__, kwargs))
                    kernel_kwargs.pop('potential_fn', None)
                    nchains = self.nwalkers
                    kernel_name = 'aies' if self.sampler == 'emcee' else 'ess'

                # Initial positions:
                if self.data.starting_point is not None:
                    initial_position = np.array([self.data.starting_point[pname] for pname in self.paramnames], dtype=float)
                    if self.sampler == 'nuts':
                        initial_x = initial_position + np.zeros((nchains, self.nparams))
                    else:
                        # Perturb initial position for each of the walkers:
                        initial_x = initial_position + self.emcee_factor * np.asarray(
                                    jax.random.normal(init_key, (nchains, self.nparams)))
                    initial_z = self._z_of_x(jnp.asarray(initial_x))
                else:
                    if self.data.verbose:
                        print('\t No starting_point given; initializing chains at the best of a set of prior draws.')
                    initial_z = self._best_prior_draws(init_key, nchains)

                # (ensemble samplers vmap the potential over walkers: evaluate it in memory-bounded chunks)
                potential = jax.jit(self._potential_z) if self.sampler == 'nuts' else chunked_vmap(jax.jit(self._potential_z), self.batch_size)
                samples_z, _ = run_numpyro(kernel_name, potential, initial_z, run_key,
                                           num_warmup=self.nburnin, num_samples=self.nsteps, num_chains=nchains,
                                           kernel_kwargs=kernel_kwargs, mcmc_kwargs=mcmc_kwargs)

                x_of_z = jax.jit(jax.vmap(jax.vmap(self._x_of_z)))
                chains = np.asarray(x_of_z(jnp.asarray(samples_z)))

                # Store posterior samples. First, store the samples for each walker/chain (shape (steps, walkers, parameters), as emcee):
                out['posteriors_per_walker'] = np.swapaxes(chains, 0, 1)

                # And now store posteriors with all walkers flattened out:
                posterior_samples = chains.reshape(-1, self.nparams)

            # Save posterior samples as outputted by the sampler:
            out['posterior_samples'] = {}
            out['posterior_samples']['unnamed'] = posterior_samples

            # Save log-likelihood of each of the samples:
            out['posterior_samples']['loglike'] = np.asarray(jax.lax.map(self._loglike_x_jit, jnp.asarray(posterior_samples),
                                                                         batch_size=min(len(posterior_samples), self.batch_size)))

            pcounter = 0
            for pname in self.model_parameters:
                if data.priors[pname]['distribution'].lower() != 'fixed':
                    self.posteriors[pname] = np.median(
                        posterior_samples[:, pcounter])
                    out['posterior_samples'][
                        pname] = posterior_samples[:, pcounter]
                    pcounter += 1

            # Go through the posterior samples to see if dt or T, the TTV parameters, are present. If they are, add to the posterior dictionary
            # (.pkl) and file (.dat) the corresponding time-of-transit center, if the dt parametrization is being used, which is the actual
            # observable folks doing dynamics usually want. If the T parametrization is being used, write down the period and t0 implied by
            # those T's:
            fitted_parameters = list(out['posterior_samples'].keys())
            firstTime, Tparametrization = True, False
            for posterior_parameter in fitted_parameters:
                pvector = posterior_parameter.split('_')
                if pvector[0] == 'dt':
                    # Extract planet number (pnum, e.g., 'p1'), instrument (ins, e.g., 'TESS') and transit number (tnum, e.g., '-1'):
                    pnum, ins, tnum = pvector[1:]
                    # Extract the period; check if it was fitted. If not, assume it was fixed:
                    if 'P_' + pnum in fitted_parameters:
                        P = out['posterior_samples']['P_' + pnum]
                    else:
                        P = data.priors['P_' + pnum]['hyperparameters']
                    # Same for t0:
                    if 't0_' + pnum in fitted_parameters:
                        t0 = out['posterior_samples']['t0_' + pnum]
                    else:
                        t0 = data.priors['t0_' + pnum]['hyperparameters']
                    # Having extracted P and t0, generate the time-of-transit center for the current transit:
                    out['posterior_samples'][
                        'T_' + pnum + '_' + ins + '_' +
                        tnum] = t0 + np.double(tnum) * P + out[
                            'posterior_samples'][posterior_parameter]
                if pvector[0] == 'T':
                    if firstTime:
                        Tparametrization = True
                        Tdict = {}
                        firstTime = False
                    # Extract planet number (pnum, e.g., 'p1'), instrument (ins, e.g., 'TESS') and transit number (tnum, e.g., '-1'):
                    pnum, ins, tnum = pvector[1:]
                    if pnum not in list(Tdict.keys()):
                        Tdict[pnum] = {}
                    Tdict[pnum][int(
                        tnum)] = out['posterior_samples'][posterior_parameter]
            if Tparametrization:
                for pnum in list(Tdict.keys()):
                    all_ns = np.array(list(Tdict[pnum].keys()))
                    all_Ts = np.array([Tdict[pnum][n] for n in all_ns])
                    N = len(all_ns)
                    XY, Y, X, X2 = np.sum(all_Ts * all_ns[:, None], axis=0) / N, np.sum(all_Ts, axis=0) / N, \
                                   np.sum(all_ns) / N, np.sum(all_ns**2) / N
                    # Get slope and intercept:
                    out['posterior_samples']['P_' + pnum] = (XY - X * Y) / (X2 - (X**2))
                    out['posterior_samples']['t0_' + pnum] = Y - out['posterior_samples']['P_' + pnum] * X
            if self.data.t_lc is not None:
                if True in self.data.lc_options['efficient_bp'].values():
                    out['pu'] = self.pu
                    out['pl'] = self.pl
            if self.data.t_rv is not None:
                if self.data.rv_options['fitrvline'] or self.data.rv_options[
                        'fitrvquad']:
                    out['ta'] = self.ta

            # Finally, save juliet output to pickle file:
            pickle.dump(
                out,
                open(self.out_folder + self.sampler_prefix + 'posteriors.pkl',
                     'wb'))
        else:
            # If the sampler was already ran, then user really wants to extract outputs from previous fit:
            print('Detected ' + self.sampler +
                  ' sampler output files --- extracting from ' +
                  self.out_folder + self.sampler_prefix + 'posteriors.pkl')
            if self.data.pickle_encoding is None:
                out = pickle.load(
                    open(
                        self.out_folder + self.sampler_prefix +
                        'posteriors.pkl', 'rb'))
            else:
                out = pickle.load(open(
                    self.out_folder + self.sampler_prefix + 'posteriors.pkl',
                    'rb'),
                                  encoding=self.data.pickle_encoding)
            if len(out.keys()) == 0:
                print(
                    'Warning: no output generated or extracted. Check the fit options given to juliet.fit().'
                )
            else:
                # For retro-compatibility, check for sigma_w_rv_instrument and add an extra variable on out
                # for sigma_w_instrument:
                out_temp = dict()
                for pname in out['posterior_samples'].keys():
                    if 'sigma_w_rv' == pname[:10]:
                        instrument = pname.split('_')[-1]
                        out_temp['sigma_w_' + instrument] = out['posterior_samples'][pname]
                for pname in out_temp.keys():
                    out['posterior_samples'][pname] = out_temp[pname]
                # Extract parameters:
                for pname in self.posteriors.keys():
                    if data.priors[pname]['distribution'].lower() != 'fixed':
                        self.posteriors[pname] = np.median(
                            out['posterior_samples'][pname])
                posterior_samples = out['posterior_samples']['unnamed']
                if 'pu' in out.keys():
                    self.pu = out['pu']
                    self.pl = out['pl']
                    self.Ar = (self.pu - self.pl) / (2. + self.pl + self.pu)
                if 'ta' in out.keys():
                    self.ta = out['ta']

        # Either fit done or extracted. If doesn't exist, create the posteriors.dat file:
        if self.out_folder is not None:
            if not os.path.exists(self.out_folder + 'posteriors.dat'):
                outpp = open(self.out_folder + 'posteriors.dat', 'w')
                writepp(outpp, out, data.priors)

        # Save all results (posteriors) to the self.results object:
        self.posteriors = out

        # Save posteriors to lc and rv:
        if self.data.t_lc is not None:
            self.lc.set_posterior_samples(out['posterior_samples'])
        if self.data.t_rv is not None:
            self.rv.set_posterior_samples(out['posterior_samples'])


def _quantiles(samples, alpha=0.68):
    """Vectorized version of juliet.utils.get_quantiles along the first axis of samples."""
    ordered = np.sort(samples, axis=0)
    nsamples = samples.shape[0]
    # (capped so that the indices below stay within the array for small numbers of samples)
    nsamples_at_each_side = min(int(nsamples * (alpha / 2.) + 1), nsamples // 2 - 2 if nsamples % 2 == 0 else (nsamples - 1) // 2)
    if nsamples % 2 == 0:
        med_idx_up = int(nsamples / 2.) + 1
        med_idx_down = med_idx_up - 1
        return (ordered[med_idx_up] + ordered[med_idx_down]) / 2., ordered[med_idx_up + nsamples_at_each_side], \
               ordered[med_idx_down - nsamples_at_each_side]
    med_idx = int(nsamples / 2.)
    return ordered[med_idx], ordered[med_idx + nsamples_at_each_side], ordered[med_idx - nsamples_at_each_side]


class model(object):
    """
    Given a juliet data object, this kernel generates either a lightcurve or a radial-velocity object. Example usage:

               >>> model = juliet.model(data, modeltype = 'lc')

    Models are pure JAX functions of a dictionary of parameter values: lightcurves are computed with jaxoplanet
    (Kepler solver plus polynomial limb-darkening for the linear and quadratic laws; other laws and catwoman-like
    asymmetric transits are integrated numerically) and RVs with jaxoplanet's Keplerian system.

    :param data: (juliet.load object)
        An object containing all the information about the current dataset.

    :param modeltype: (optional, string)
        String indicating whether the model to generate should be a lightcurve ('lc') or a radial-velocity ('rv') model.

    :param pl: (optional, float)
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled.
        Default is ``0``.

    :param pu: (optional, float)
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ecclim: (optional, float)
        This parameter sets the maximum eccentricity allowed such that a model is actually evaluated. Default is ``1``.

    :param light_travel_delay: (optinal, bool)
        Boolean indicating if light travel time delay wants to be included on eclipse time calculations.

    :param stellar_radius: (optional, float)
        Stellar radius in units of solar-radii to use for the light travel time corrections.

    :param log_like_calc: (optional, boolean)
        Kept for back-compatibility; it has no effect.

    """

    def __new__(cls, data, *args, **kwargs):
        # Data loaded with backend = 'legacy' are handled by the legacy implementation:
        if getattr(data, 'backend', 'jax') == 'legacy':
            from .legacy.fit import model as legacy_class
            return legacy_class(data, *args, **kwargs)
        return super().__new__(cls)

    ###########################################################################################
    # Pure (JAX) model functions. ``pv`` is a dictionary with the values of all the parameters.
    ###########################################################################################

    def _ld_coefficients(self, pv, instrument):
        ldlaw = self.dictionary[instrument]['ldlaw']
        parametrization = self.dictionary[instrument]['ldparametrization']
        if ldlaw == 'none':
            return jnp.array([0.1, 0.3])
        name = self.ld_iname[instrument]
        if ldlaw == 'linear':
            if parametrization == 'kipping2013':
                return jnp.stack([pv['q1_' + name]])
            return jnp.stack([pv['u1_' + name]])
        if ldlaw == 'nonlinear':
            return jnp.stack([pv['c' + str(k) + '_' + name] for k in range(1, 5)])
        if parametrization == 'kipping2013':
            coeff1, coeff2 = jm.reverse_ld_coeffs(ldlaw, pv['q1_' + name], pv['q2_' + name])
        else:
            coeff1, coeff2 = pv['u1_' + name], pv['u2_' + name]
        return jnp.stack([coeff1, coeff2])

    def _T_parametrization_ephemerides(self, pv):
        # If TTV parametrization is 'T' for planet i, the period and t0 are the least-squares slope and intercept of the
        # transit times of all instruments:
        planet_t0, planet_P = {}, {}
        for i in self.numbering:
            if self.Tparametrization.get(i, False):
                all_Ts, all_ns = [], []
                for instrument in self.all_inames:
                    for transit_number in self.dictionary[instrument]['TTVs'][int(i)]['transit_number']:
                        all_Ts.append(pv['T_p' + str(i) + '_' + instrument + '_' + str(transit_number)])
                        all_ns.append(transit_number)
                all_Ts, all_ns = jnp.stack(all_Ts), np.array(all_ns, dtype=float)
                N = self.N_TTVs[i]
                XY, Y, X, X2 = jnp.sum(all_Ts * all_ns) / N, jnp.sum(all_Ts) / N, np.sum(all_ns) / N, np.sum(all_ns**2) / N
                planet_P[i] = (XY - X * Y) / (X2 - (X**2))
                planet_t0[i] = Y - planet_P[i] * X
        return planet_t0, planet_P

    def _ecc_omega(self, pv, i):
        """Eccentricity and omega (in radians) of planet i."""
        si = str(i)
        if self.dictionary['ecc_parametrization'][i] == 0:
            return pv['ecc_p' + si], pv['omega_p' + si] * np.pi / 180.
        elif self.dictionary['ecc_parametrization'][i] == 1:
            ecosw, esinw = pv['ecosomega_p' + si], pv['esinomega_p' + si]
            return jm.safe_sqrt(ecosw**2 + esinw**2), jnp.arctan2(esinw, ecosw)
        secosw, sesinw = pv['secosomega_p' + si], pv['sesinomega_p' + si]
        return secosw**2 + sesinw**2, jnp.arctan2(sesinw, secosw)

    def _planet_lightcurve(self, pv, instrument, i, P, t0, t):
        """Lightcurve of planet i for instrument at (TTV-shifted) times t; returns (flux, ok)."""
        d = self.dictionary[instrument]
        si = str(i)
        pkey = 'p' + si

        if self.dictionary['efficient_bp'][i]:
            if not self.dictionary['fitrho']:
                a = pv['a_p' + si]
            else:
                a = ((pv['rho'] * G * ((P * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)
            r1, r2 = pv['r1_p' + si], pv['r2_p' + si]
            sq = jm.safe_sqrt(r1 / self.Ar)
            b = jnp.where(r1 > self.Ar, (1 + self.pl) * (1. + (r1 - 1.) / (1. - self.Ar)),
                          (1. + self.pl) + sq * r2 * (self.pu - self.pl))
            p = jnp.where(r1 > self.Ar, (1 - r2) * self.pl + r2 * self.pu,
                          self.pu + (self.pl - self.pu) * sq * (1. - r2))
        else:
            if not self.dictionary['fitrho']:
                a = pv['a_p' + si]
            else:
                a = ((pv['rho'] * G * ((P * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)
            if not d['TransitFitCatwoman']:
                b, p = pv['b_p' + si], pv['p_p' + si + self.p_iname[pkey][instrument]]
            else:
                # Planet made of two semicircles of radii p1 and p2 rotated by an angle phi (catwoman):
                b = pv['b_p' + si]
                p1 = pv['p1_p' + si + self.p1_iname[pkey][instrument]]
                p2 = pv['p2_p' + si + self.p1_iname[pkey][instrument]]
                phi = pv['phi_p' + si]
                p = jnp.minimum(p1, p2)

        ecc, omega = self._ecc_omega(pv, i)
        ok = ecc <= self.ecclim
        # Safe values so that invalid regions of parameter space don't produce NaNs (or NaN gradients):
        ecc = jnp.where(ecc < 1., ecc, 0.)
        sinw, cosw = jnp.sin(omega), jnp.cos(omega)
        ecc_factor = (1. + ecc * sinw) / (1. - ecc**2)
        cosi = (b / a) * ecc_factor
        ok = ok & jnp.logical_not((b > 1. + p) | (cosi >= 1.))
        cosi = jnp.where(cosi < 1., cosi, 0.)
        sini = jnp.sqrt(1. - cosi**2)

        if d['resampling']:
            nresampling, etresampling = d['nresampling'], d['exptimeresampling']
        else:
            nresampling, etresampling = None, None

        tp = jm.time_of_periastron(t0, P, ecc, omega)

        if d['TransitFit'] or d['TranEclFit']:
            tt = jm.supersample(t, nresampling, etresampling)
            dd, zz, _, _ = jm.sky_position(tt, tp, P, a, ecc, sinw, cosw, sini, cosi)
            if d['TransitFitCatwoman']:
                X, Y, zz, direction = jm.sky_coordinates(tt, tp, P, a, ecc, sinw, cosw, sini, cosi)
                transit_model = jm.semicircles_transit_flux(X, Y, zz, direction, p1, p2, phi * np.pi / 180., d['ldlaw'],
                                                            self._ld_coefficients(pv, instrument))
            elif d['ldlaw'] in jm.POLYNOMIAL_LD_LAWS:
                transit_model = jm.transit_flux(dd, zz, p, self._ld_coefficients(pv, instrument))
            else:
                transit_model = jm.numerical_transit_flux(dd, zz, p, d['ldlaw'], self._ld_coefficients(pv, instrument))
            transit_model = transit_model.mean(axis=1)
            if d['TransitFit']:
                return transit_model, ok

        # Eclipse (with or without transit):
        fpname = 'fp_p' + si + self.fp_iname[pkey].get(instrument, '')
        # There is no fp prior for, e.g., Lambertian or kelp phase curves; use a dummy value:
        fp = pv[fpname] if fpname in pv else 100e-6
        if self.light_travel_delay:
            # Self-consistently calculate time of secondary eclipse and light-travel corrected times:
            t_secondary = jm.time_of_secondary(t0, P, ecc, omega)
            tp_eclipse = tp
            eclipse_t = jm.light_travel_corrected_times(t, t0, tp, P, a, ecc, sinw, cosw, sini, self.stellar_radius)
        else:
            t_secondary = pv['t_secondary_p' + si]
            tp_eclipse = jm.time_of_periastron(t_secondary, P, ecc, omega, secondary=True)
            eclipse_t = t
        tt = jm.supersample(eclipse_t, nresampling, etresampling)
        dd, zz, _, _ = jm.sky_position(tt, tp_eclipse, P, a, ecc, sinw, cosw, sini, cosi)
        eclipse_model = jm.eclipse_flux(dd, zz, p, fp).mean(axis=1)

        if d['EclipseFit']:
            return eclipse_model, ok

        # Combined transit + eclipse models. Assume by default any phase-curve variations are being modelled externally
        # (by, e.g., systematics models). Note out-of-eclipse the eclipse model is 1 + fp --- with in-eclipse always being 1.
        # Subtract fp then:
        if not self.phase_curve[instrument]:
            return transit_model * (eclipse_model - fp), ok

        # Otherwise, add all the phase curve models (evaluated at the non-supersampled eclipse times):
        phase_curve_model = jnp.zeros_like(eclipse_t)
        # Orbital phases of the sinusoidal and kelp phase curves are measured from the time of transit implied by the
        # secondary eclipse ephemeris (as in previous juliet versions, where batman set it from t_secondary), so they
        # are aligned with the secondary eclipse:
        t0_eclipse = jm.time_of_conjunction(t_secondary, P, ecc, omega)
        if d['PhaseCurveFit']:
            phase_offset = pv['phaseoffset_p' + si + self.phaseoffset_iname[pkey][instrument]]
            phase_curve_model = phase_curve_model + jm.sinusoidal_phase_curve(eclipse_t, t0_eclipse, P, fp, phase_offset)
        if d['CowanAgolPCFit']:
            suffix = si + self.fp_iname[pkey][instrument]
            phase_curve_model = phase_curve_model + jm.cowan_agol_phase_curve(eclipse_t, t_secondary, P, fp,
                                    pv['C1_p' + suffix], pv['D1_p' + suffix], pv['C2_p' + suffix], pv['D2_p' + suffix])
        if d['LambertPCFit']:
            sinf, cosf = jm.true_anomaly(eclipse_t, tp_eclipse, P, ecc)
            Ag_Lambert = pv['aglambert_p' + si + self.aglambert_iname[pkey][instrument]]
            phase_curve_model = phase_curve_model + jm.lambertian_phase_curve(sinf, cosf, sinw, cosw, sini, ecc, p, a, Ag_Lambert)
        if d['KelpHomoPCFit']:
            suffix = si + self.kelphomo_iname[pkey][instrument]
            phase_curve_model = phase_curve_model + kelp_homogeneous_refl_pc_model(times=eclipse_t, t0=t0_eclipse, per=P, ar=a, rprs=p,
                                    g=pv['g_p' + suffix], single_scat_albedo=pv['singlescat_p' + suffix],
                                    nknots=self.kelp_refl_interpolation_knots)
        if d['KelpThmPCFit']:
            suffix = si + self.kelpthm_iname[pkey][instrument]
            phase_curve_model = phase_curve_model + kelp_thermal_pc_model(times=eclipse_t, t0=t0_eclipse, per=P, ar=a, rprs=p,
                                    filter_wavelength=self.kelp_filt_wav[instrument],
                                    filter_transmittance=self.kelp_filt_trans[instrument],
                                    hotspot_offset=pv['hotspotoff_p' + suffix], c11=pv['cml11_p' + suffix],
                                    fprime=pv['fprime_p' + suffix], alpha=pv['alpha_p' + suffix],
                                    omega_drag=pv['wdrag_p' + suffix], Teff=self.stellar_teff,
                                    ntheta=self.kelp_ntheta, nphi=self.kelp_nphi,
                                    nknots=self.kelp_thm_interpolation_knots)
        if d['KelpInhomoPCFit']:
            suffix = si + self.kelpinhomo_iname[pkey][instrument]
            if 'x1_p' + suffix in pv:
                x1, x2 = pv['x1_p' + suffix], pv['x2_p' + suffix]
            else:
                # x1' = sin(x1), x2' = sin(x2) (Morris et al. 2024):
                x1, x2 = jnp.rad2deg(jnp.arcsin(pv['x1prime_p' + suffix])), jnp.rad2deg(jnp.arcsin(pv['x2prime_p' + suffix]))
            phase_curve_model = phase_curve_model + kelp_inhomogeneous_refl_pc_model(times=eclipse_t, t0=t0_eclipse, per=P, ar=a, rprs=p,
                                    w0=pv['w0_p' + suffix], wp=pv['wp_p' + suffix], Ag=pv['agkelp_p' + suffix], x1=x1, x2=x2,
                                    nknots=self.kelp_refl_interpolation_knots)

        # Multiply the phase curve with the normalized occultation model:
        phase_curve_model = 1. + phase_curve_model * ((eclipse_model - 1.) / fp)
        return transit_model * phase_curve_model, ok

    def _lc_instrument_model(self, pv, instrument, t, lm_arguments=None, include_lm=True, at_data_times=True):
        """Lightcurve model of an instrument at times t. Returns a dictionary with the model components and an 'ok' flag."""
        d = self.dictionary[instrument]
        out = {}
        ok = jnp.array(True)
        M = jnp.ones_like(t)

        if d['TransitFit'] or d['EclipseFit'] or d['TranEclFit']:

            if self.Tflag:
                planet_t0, planet_P = self._T_parametrization_ephemerides(pv)

            cP, ct0, ctimes = {}, {}, {}
            for i in self.numbering:
                ttv = d['TTVs'][i]
                dummy_time = t
                if not ttv['status']:
                    P, t0 = pv['P_p' + str(i)], pv['t0_p' + str(i)]
                elif ttv['parametrization'] == 'dt':
                    # If, e.g., dt_p1_TESS1_-2, then n = -2 and the time of transit (with TTV) = t0 + n*P + dt_p1_TESS1_-2. This
                    # implicitly sets maximum transit duration to P/2 days:
                    P, t0 = pv['P_p' + str(i)], pv['t0_p' + str(i)]
                    for transit_number in ttv['transit_number']:
                        dt = pv['dt_p' + str(i) + '_' + instrument + '_' + str(transit_number)]
                        transit_time = t0 + transit_number * P + dt
                        dummy_time = jnp.where(jnp.abs(t - transit_time) < P / 4., t - dt, dummy_time)
                else:
                    t0, P = planet_t0[i], planet_P[i]
                    for transit_number in ttv['transit_number']:
                        transit_time = pv['T_p' + str(i) + '_' + instrument + '_' + str(transit_number)]
                        dt = transit_time - (t0 + transit_number * P)
                        dummy_time = jnp.where(jnp.abs(t - transit_time) < P / 4., t - dt, dummy_time)
                cP[i], ct0[i], ctimes[i] = P, t0, dummy_time

            # Check the periods are chronologically ordered (this is to avoid multiple modes due to periods "jumping" between
            # planet numbering):
            for i, j in zip(self.numbering[:-1], self.numbering[1:]):
                ok = ok & (cP[i] < cP[j])

            for i in self.numbering:
                flux, planet_ok = self._planet_lightcurve(pv, instrument, i, cP[i], ct0[i], ctimes[i])
                ok = ok & planet_ok
                out['p' + str(i)] = flux
                M = M + flux - 1.

        # Convert the lightcurve so it complies with the juliet model accounting for the dilution and the mean out-of-transit flux:
        D, Mflux = pv['mdilution_' + self.mdilution_iname[instrument]], pv['mflux_' + self.mflux_iname[instrument]]
        M = (M * D + (1. - D)) * (1. / (1. + D * Mflux))
        out['M'] = M

        deterministic = M
        if include_lm and self.lm_boolean[instrument]:
            out['LM'] = self._linear_model(pv, instrument, lm_arguments)
            deterministic = deterministic + out['LM']

        if self.nlm_boolean[instrument]:
            if not at_data_times:
                raise Exception('Non-linear functions can only be evaluated at the times of the data.')
            out['NLM'] = self.nlm_functions[instrument](pv)
            if self.multiplicative_non_linear_function[instrument]:
                deterministic = deterministic * out['NLM']
            else:
                deterministic = deterministic + out['NLM']

        out['deterministic'] = deterministic
        out['ok'] = ok
        return out

    def _linear_model(self, pv, instrument, lm_arguments):
        LM = 0.
        for i in range(self.lm_n[instrument]):
            LM = LM + pv['theta' + str(i) + '_' + self.theta_iname[str(i) + instrument]] * lm_arguments[:, i]
        return LM

    def _rv_keplerian(self, pv, t):
        """Keplerian (plus trend) RV model at times t; returns per-planet keplerians, full keplerian, trend and an 'ok' flag."""
        ok = jnp.array(True)
        # Check the periods are chronologically ordered:
        for i, j in zip(self.numbering[:-1], self.numbering[1:]):
            ok = ok & (pv['P_p' + str(i)] < pv['P_p' + str(j)])
        out = {}
        keplerian = jnp.zeros_like(t)
        for i in self.numbering:
            si = str(i)
            ecc, omega = self._ecc_omega(pv, i)
            ok = ok & (ecc <= self.ecclim)
            ecc = jnp.where(ecc < 1., ecc, 0.)
            out['p' + si] = jm.rv_keplerian(t, pv['P_p' + si], pv['t0_p' + si], ecc, omega, pv['K_p' + si])
            keplerian = keplerian + out['p' + si]
        if self.dictionary['fitrvline']:
            trend = pv['rv_intercept'] + (t - self.ta) * pv['rv_slope']
        elif self.dictionary['fitrvquad']:
            trend = pv['rv_intercept'] + (t - self.ta) * pv['rv_slope'] + ((t - self.ta)**2) * pv['rv_quad']
        else:
            trend = jnp.zeros_like(t)
        out['Keplerian'] = keplerian
        out['trend'] = trend
        out['ok'] = ok
        return out

    def _rv_instrument_model(self, pv, instrument, t, lm_arguments=None, include_lm=True, at_data_times=True):
        out = self._rv_keplerian(pv, t)
        out['M'] = out['Keplerian'] + out['trend'] + pv['mu_' + instrument]
        deterministic = out['M']
        if include_lm and self.lm_boolean[instrument]:
            out['LM'] = self._linear_model(pv, instrument, lm_arguments)
            deterministic = deterministic + out['LM']
        out['deterministic'] = deterministic
        return out

    def _instrument_model(self, pv, instrument, t=None, lm_arguments=None, include_lm=True):
        if t is None:
            t, lm_arguments, at_data_times = self.jtimes[instrument], self.jlm_arguments.get(instrument), True
        else:
            t, at_data_times = jnp.asarray(t, dtype=float), False
            if lm_arguments is not None:
                lm_arguments = jnp.asarray(lm_arguments, dtype=float)
        if self.modeltype == 'lc':
            return self._lc_instrument_model(pv, instrument, t, lm_arguments, include_lm, at_data_times)
        return self._rv_instrument_model(pv, instrument, t, lm_arguments, include_lm, at_data_times)

    def _variances(self, pv, instrument):
        if self.modeltype == 'lc':
            return self.jerrors[instrument]**2 + (pv['sigma_w_' + self.sigmaw_iname[instrument]] * 1e-6)**2
        return self.jerrors[instrument]**2 + pv['sigma_w_' + instrument]**2

    def _global_model(self, pv, include_lm=True):
        """Deterministic model and variances of all instruments, stacked in the global data arrays."""
        deterministic = jnp.zeros(len(self.t))
        variances = jnp.zeros(len(self.t))
        ok = jnp.array(True)
        outs = {}
        for instrument in self.inames:
            out = self._instrument_model(pv, instrument, include_lm=include_lm)
            idx = self.instrument_indexes[instrument]
            deterministic = deterministic.at[idx].set(out['deterministic'])
            variances = variances.at[idx].set(self._variances(pv, instrument))
            ok = ok & out['ok']
            outs[instrument] = out
        return deterministic, variances, ok, outs

    def log_likelihood_fn(self, pv):
        """Log-likelihood of the data given the parameter dictionary pv (pure JAX function)."""
        if self.global_model:
            deterministic, variances, ok, _ = self._global_model(pv)
            residuals = self.jy - deterministic
            if self.dictionary['global_model']['GPDetrend']:
                log_like = self.dictionary['global_model']['noise_model'].log_likelihood(pv, residuals, variances)
            else:
                log_like = jm.gaussian_log_likelihood(residuals, variances)
        else:
            log_like = 0.
            ok = jnp.array(True)
            for instrument in self.inames:
                out = self._instrument_model(pv, instrument)
                ok = ok & out['ok']
                residuals = self.jdata[instrument] - out['deterministic']
                variances = self._variances(pv, instrument)
                if self.dictionary[instrument]['GPDetrend']:
                    log_like = log_like + self.dictionary[instrument]['noise_model'].log_likelihood(pv, residuals, variances)
                else:
                    log_like = log_like + jm.gaussian_log_likelihood(residuals, variances)
        return jnp.where(ok, log_like, -jnp.inf)

    def _parameter_dictionary(self, parameter_values):
        pv = {p: jnp.asarray(v, dtype=float) for p, v in self.fixed_values.items()}
        for p, v in parameter_values.items():
            if p != 'unnamed':
                pv[p] = jnp.asarray(v, dtype=float)
        return pv

    def get_log_likelihood(self, parameter_values):
        return float(self._log_likelihood_jit(self._parameter_dictionary(parameter_values)))

    ###########################################################################################
    # Model generation and evaluation
    ###########################################################################################

    def _generate_fn(self, pv):
        outs = {}
        if self.global_model:
            outs['global'], outs['global_variances'], ok, per_instrument = self._global_model(pv)
        else:
            ok = jnp.array(True)
            per_instrument = {}
            for instrument in self.inames:
                per_instrument[instrument] = self._instrument_model(pv, instrument)
                ok = ok & per_instrument[instrument]['ok']
        for instrument in self.inames:
            per_instrument[instrument]['deterministic_variances'] = self._variances(pv, instrument)
        outs['instruments'] = per_instrument
        if self.modeltype == 'rv':
            outs['rv'] = self._rv_keplerian(pv, jnp.asarray(self.t, dtype=float))
        outs['ok'] = ok
        return outs

    def generate_lc_model(self, parameter_values, evaluate_global_errors=True, evaluate_lc=False):
        """Evaluates the lightcurve model on the data at the given parameter values; results are saved to self.model."""
        return self._generate(parameter_values)

    def generate_rv_model(self, parameter_values, evaluate_global_errors=True):
        """Evaluates the RV model on the data at the given parameter values; results are saved to self.model."""
        return self._generate(parameter_values)

    def _generate(self, parameter_values):
        outs = jax.tree.map(np.asarray, self._generate_jit(self._parameter_dictionary(parameter_values)))
        self.modelOK = bool(outs['ok'])
        if self.global_model:
            self.model['global'] = outs['global']
            self.model['global_variances'] = outs['global_variances']
        for instrument in self.inames:
            for k, v in outs['instruments'][instrument].items():
                if k != 'ok':
                    self.model[instrument][k] = v
        if self.modeltype == 'rv':
            for i in self.numbering:
                self.model['p' + str(i)] = outs['rv']['p' + str(i)]
            self.model['Keplerian'] = outs['rv']['Keplerian']
            self.model['Keplerian+Trend'] = outs['rv']['Keplerian'] + outs['rv']['trend']
        if not self.modelOK:
            return False

    def _evaluate_single(self, pv, instrument, t, GPregressors, LMregressors, evaluate_transit):
        """Model (and its components) for a single set of parameter values. Pure JAX function."""
        include_lm = not evaluate_transit
        result = {}
        gp_on = (not evaluate_transit) and (self.dictionary['global_model']['GPDetrend'] if self.global_model
                                            else self.dictionary[instrument]['GPDetrend'])
        if self.global_model:
            deterministic, variances, _, outs = self._global_model(pv, include_lm=include_lm)
            if t is None:
                result['deterministic'] = deterministic
                evaluated = outs
            else:
                lm = None if LMregressors is None or instrument not in LMregressors else LMregressors[instrument]
                evaluated = dict(outs)
                evaluated[instrument] = self._instrument_model(pv, instrument, t, lm, include_lm=include_lm)
                result['deterministic'] = evaluated[instrument]['deterministic']
            if gp_on:
                residuals = self.jy - deterministic
                result['GP'] = self.dictionary['global_model']['noise_model'].predict(pv, residuals, variances,
                                                                                     None if t is None else GPregressors)
        else:
            outs = self._instrument_model(pv, instrument, include_lm=include_lm)
            evaluated = {instrument: outs if t is None else self._instrument_model(pv, instrument, t, LMregressors,
                                                                                  include_lm=include_lm)}
            result['deterministic'] = evaluated[instrument]['deterministic']
            if gp_on:
                residuals = self.jdata[instrument] - outs['deterministic']
                result['GP'] = self.dictionary[instrument]['noise_model'].predict(pv, residuals, self._variances(pv, instrument),
                                                                                 None if t is None else GPregressors)
        result['model'] = result['deterministic'] + result['GP'] if gp_on else result['deterministic']

        # Components of the model:
        components = {}
        component_instruments = self.inames if self.global_model else [instrument]
        for ginstrument in component_instruments:
            out = evaluated[ginstrument]
            c = {}
            for i in self.numbering:
                c['p' + str(i)] = out.get('p' + str(i), jnp.ones_like(out['M']))
            if self.modeltype == 'lc':
                c['transit'] = 1. + sum(c['p' + str(i)] - 1. for i in self.numbering)
            else:
                c['keplerian'] = out['Keplerian']
                c['trend'] = out['trend']
                c['mu'] = pv['mu_' + ginstrument]
            c['lm'] = out.get('LM', jnp.zeros_like(out['M']))
            components[ginstrument] = c
        result['components'] = components
        return result

    def evaluate_model(self, instrument = None, parameter_values = None,
                          all_samples = False, nsamples = 1000, return_samples = False, t = None, GPregressors = None, LMregressors = None,
                          return_err = False, alpha = 0.68, return_components = False, evaluate_transit = False):
        """
        This function evaluates the current lc or rv model given a set of posterior distribution samples and/or parameter values. Example usage:

                             >>> dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error)
                             >>> results = dataset.fit()
                             >>> transit_model, error68_up, error68_down = results.lc.evaluate('TESS', return_err=True)

        Or:

                             >>> dataset = juliet.load(priors=priors, t_rv = times, y_rv = fluxes, yerr_rv = fluxes_error)
                             >>> results = dataset.fit()
                             >>> rv_model, error68_up, error68_down = results.rv.evaluate('FEROS', return_err=True)

        Models for all the posterior samples are evaluated in a vectorized way with JAX.

        :param instrument: (optional, string)
        Instrument the user wants to evaluate the model on. It is expected to be given for non-global models, not necessary for global models.

        :param parameter_values: (optional, dict)
        Dictionary containing samples of the posterior distribution or, more generally, parameter valuesin it. Each key is a parameter name (e.g. 'p_p1',
        'q1_TESS', etc.), and inside each of those keys an array of N samples is expected (i.e., parameter_values['p_p1'] is an array of length N). The
        indexes have to be consistent between different parameters.

        :param all_samples: (optional, boolean)
        If True, all posterior samples will be used to evaluate the model. Default is False.

        :param nsamples: (optional, int)
        Number of posterior samples to be used to evaluate the model. Default is 1000 (note each call to this function will sample `nsamples` different samples
        from the posterior, so no two calls are exactly the same).

        :param return_samples: (optional, boolean)
        Boolean indicating whether the user wants the posterior model samples (i.e., the models evaluated in each of the posterior sample draws) to be returned. Default
        is False.

        :param t: (optional, numpy array)
        Array with the times at which the model wants to be evaluated.

        :param GPRegressors: (optional, numpy array)
        Array containing the GP Regressors onto which to evaluate the models. Dimensions must be consistent with input `t`.

        :param LMRegressors: (optional, numpy array or dictionary)
        If the model is not global, this is an array containing the Linear Regressors onto which to evaluate the model for the input instrument.
        Dimensions must be consistent with input `t`. If model is global, this needs to be a dictionary.

        :param return_err: (optional, boolean)
        If True, this returns the credibility interval on the evaluated model. Default credibility interval is 68%.

        :param alpha: (optional, double)
        Credibility interval for return_err. Default is 0.68, i.e., the 68% credibility interval.

        :param return_components: (optional, boolean)
        If True, each component of the model is returned (i.e., the Gaussian Process component, the Linear Model component, etc.).

        :param evaluate_transit: (optional, boolean)
        If True, the function evaluates only the transit model and not the Gaussian Process or Linear Model components.

        :returns: By default, the function returns the median model as evaluated with the posterior samples. Depending on the options chosen by the user, this can return up to 5 elements (in that order): `model_samples`, `median_model`, `upper_CI`, `lower_CI` and `components`. The first is an array with all the model samples as evaluated from the posterior. The second is the median model. The third and fourth are the uppper and lower Credibility Intervals, and the latter is a dictionary with the model components.

        """
        if evaluate_transit and self.modeltype != 'lc':
            raise Exception(
                "Trying to evaluate a transit (evaluate_transit = True) in a non-lightcurve model is not allowed."
            )

        # If no instrument is given, assume user wants a global model evaluation:
        if instrument is None:
            if not self.global_model:
                raise Exception(
                    "Input error: an instrument has to be defined for non-global models in order to evaluate the model."
                )

        if t is not None:
            gp_on = (self.dictionary['global_model']['GPDetrend'] if self.global_model else self.dictionary[instrument]['GPDetrend'])
            if gp_on and (not evaluate_transit) and GPregressors is None:
                raise Exception("\t Model for instrument " + str(instrument) +
                                " has a GP, and requires a GPregressors to be inputted to be evaluated.")
            if self.global_model and LMregressors is not None:
                LMregressors = {k: jnp.asarray(v, dtype=float) for k, v in LMregressors.items()}

        if parameter_values is None:
            parameter_values = self.posteriors

        input_parameters = [p for p in parameter_values.keys() if p not in ['unnamed', 'loglike']]

        def single(pv):
            return self._evaluate_single(pv, instrument, t, GPregressors, LMregressors, evaluate_transit)

        if type(parameter_values[input_parameters[0]]) is np.ndarray:
            # To generate a median model first generate an output_model_samples array that will save the model at each evaluation. This will
            # save nsamples samples of the posterior model. If all_samples = True, all samples from the posterior are used for the evaluated model:
            nsampled = len(parameter_values[input_parameters[0]])
            if all_samples:
                nsamples = nsampled
                idx_samples = np.arange(nsamples)
            else:
                idx_samples = np.random.choice(np.arange(nsampled),
                                               np.min([nsamples, nsampled]),
                                               replace=False)
                idx_samples = idx_samples[np.argsort(idx_samples)]

            batch = self._parameter_dictionary({p: np.asarray(parameter_values[p])[idx_samples] for p in input_parameters})
            for p in self.fixed_values:
                batch[p] = jnp.full(len(idx_samples), self.fixed_values[p])

            # Evaluate all samples (in batches, to keep the memory footprint bounded):
            results = jax.tree.map(np.asarray, jax.lax.map(jax.jit(single), batch, batch_size=min(len(idx_samples), self.batch_size)))

            output_model_samples = results['model']
            gp_on = 'GP' in results
            if return_err:
                m_output_model, u_output_model, l_output_model = _quantiles(output_model_samples, alpha=alpha)
            else:
                output_model = np.nanmedian(output_model_samples, axis=0)

            # Save the deterministic and GP parts of the model (as in previous juliet versions):
            if gp_on:
                target = self.model if self.global_model else self.model[instrument]
                if return_err:
                    target['deterministic'], target['deterministic_uerror'], target['deterministic_lerror'] = \
                        _quantiles(results['deterministic'], alpha=alpha)
                    target['GP'], target['GP_uerror'], target['GP_lerror'] = _quantiles(results['GP'], alpha=alpha)
                else:
                    target['deterministic'] = np.nanmedian(results['deterministic'], axis=0)
                    target['GP'] = np.nanmedian(results['GP'], axis=0)

            if return_components:
                components = jax.tree.map(lambda x: np.median(x, axis=0), results['components'])
        else:
            result = jax.tree.map(np.asarray, jax.jit(single)(self._parameter_dictionary(
                {p: parameter_values[p] for p in input_parameters})))
            output_model = result['model']
            if 'GP' in result:
                self.model['deterministic'], self.model['GP'] = result['deterministic'], result['GP']
            if return_components:
                components = result['components']

        if return_components:
            # Non-global models return the components of the requested instrument; global models one dictionary per instrument:
            if not self.global_model:
                components = components[instrument]
            else:
                components = {k: {ginstrument: components[ginstrument][k] for ginstrument in components}
                              for k in components[self.inames[0]]}

        if return_samples:
            if return_err:
                if return_components:
                    return output_model_samples, m_output_model, u_output_model, l_output_model, components
                else:
                    return output_model_samples, m_output_model, u_output_model, l_output_model
            else:
                if return_components:
                    return output_model_samples, output_model, components
                else:
                    return output_model_samples, output_model
        else:
            if return_err:
                if return_components:
                    return m_output_model, u_output_model, l_output_model, components
                else:
                    return m_output_model, u_output_model, l_output_model
            else:
                if return_components:
                    return output_model, components
                else:
                    return output_model

    def set_posterior_samples(self, posterior_samples):
        self.posteriors = posterior_samples
        self.median_posterior_samples = {}
        for parameter in self.posteriors.keys():

            if parameter != 'unnamed':
                self.median_posterior_samples[parameter] = np.median(self.posteriors[parameter])

        for parameter in self.priors:
            if self.priors[parameter]['distribution'].lower() == 'fixed':
                self.median_posterior_samples[parameter] = self.priors[
                    parameter]['hyperparameters']
        try:
            self.generate(self.median_posterior_samples)
        except:
            print(
                'Warning: model evaluated at the posterior median did not compute properly.'
            )

    def _check_jax_support(self):
        """Raise an error for features of previous juliet versions that the JAX backend does not support."""
        for instrument in self.inames:
            d = self.dictionary[instrument]
            if d.get('TransitFitCatwoman', False) and (d.get('EclipseFit', False) or d.get('TranEclFit', False)):
                raise NotImplementedError('Instrument ' + instrument + ': catwoman (asymmetric) transits can not be combined with '
                                          'eclipses.')
            if d.get('TransitFitCatwoman', False) and any(self.dictionary['efficient_bp'].values()):
                raise NotImplementedError('catwoman (asymmetric) transits require the (b, p1, p2) parametrization, not (r1, r2).')
            if (d.get('TransitFit', False) or d.get('TranEclFit', False)) and d['ldlaw'] not in jm.SUPPORTED_LD_LAWS:
                raise NotImplementedError('Instrument ' + instrument + ': limb-darkening law "' + d['ldlaw'] + '" is not available '
                                          '(available laws: ' + ', '.join(jm.SUPPORTED_LD_LAWS[:2] + jm.NUMERICAL_LD_LAWS) + ').')

    def __init__(self,
                 data,
                 modeltype,
                 pl=0.0,
                 pu=1.0,
                 ecclim=1.,
                 ta=2458460.,
                 light_travel_delay = False,
                 stellar_radius = None,
                 kelp_refl_interpolation_knots = None,
                 kelp_thm_interpolation_knots = None,
                 kelp_filt_wav = None,
                 kelp_filt_trans = None,
                 stellar_teff = None,
                 kelp_ntheta = 5,
                 kelp_nphi = 75,
                 log_like_calc=False):
        # Inhert the priors dictionary from data:
        self.priors = data.priors
        self.fixed_values = {p: float(self.priors[p]['hyperparameters']) for p in self.priors
                             if self.priors[p]['distribution'].lower() == 'fixed'}
        # Define the ecclim value:
        self.ecclim = ecclim
        # Define ta:
        self.ta = ta
        # Define light travel time option:
        self.light_travel_delay = light_travel_delay
        if self.light_travel_delay and (stellar_radius is None):

            raise Exception('Error: if light_travel_delay is activated, a stellar radius needs to be given as well via stellar_radius = yourvalue; e.g., dataset.fit(..., light_travel_delay = True, stellar_radius = 1.1234).')

        self.stellar_radius = stellar_radius

        # This is for kelp models: user can, instead of computing kelp models for all phases in the data, choose to
        # compute kelp models for some grid values, and then interpolate over all phases to save time. We don't do this by default
        # User can turn this feature on by providing number of knots for interpolation (separate parameters for reflective and thermal phase curve)
        self.kelp_refl_interpolation_knots = kelp_refl_interpolation_knots
        self.kelp_thm_interpolation_knots = kelp_thm_interpolation_knots

        # Kelp thermal phase curve also needs transmission functions (dicts, with keys corresponding to instrument names),
        # stellar effective temperatures, and number of grid points along latitude (theta) and longitude (phi):
        self.kelp_filt_wav = kelp_filt_wav
        self.kelp_filt_trans = kelp_filt_trans
        self.stellar_teff = stellar_teff
        self.kelp_ntheta = kelp_ntheta
        self.kelp_nphi = kelp_nphi

        # Save the log_like_calc boolean:
        self.log_like_calc = log_like_calc
        # Define variable that at each iteration defines if the model is OK or not (not OK means something failed in terms of the
        # parameter space being explored):
        self.modelOK = True
        # Define a variable that will save the posterior samples:
        self.posteriors = None
        self.median_posterior_samples = None
        # Set nlm:
        self.non_linear_functions = data.non_linear_functions
        # Check if multiplicative of additive functions for each instrument:
        self.multiplicative_non_linear_function = {}
        if self.non_linear_functions is not None:

           for k in list(self.non_linear_functions.keys()):

                # (False by default for back-compatibility):
                self.multiplicative_non_linear_function[k] = bool(self.non_linear_functions[k].get('multiplicative', False))

        # Number of datapoints per instrument variable:
        self.ndatapoints_per_instrument = {}
        if modeltype == 'lc':
            self.modeltype = 'lc'
            # Inhert times, fluxes, errors, indexes, etc. from data.
            # FYI, in case this seems confusing: self.t, self.y and self.yerr save what we internally call
            # "global" data-arrays. These have the data from all the instruments stacked into an array; to recover
            # the data for a given instrument, one uses the self.instrument_indexes dictionary. On the other hand,
            # self.times, self.data and self.errors are dictionaries that on each key have the data of a given instrument.
            self.t = data.t_lc
            self.y = data.y_lc
            self.yerr = data.yerr_lc
            self.times = data.times_lc
            self.data = data.data_lc
            self.errors = data.errors_lc
            self.instruments = data.instruments_lc
            self.ninstruments = data.ninstruments_lc
            self.inames = data.inames_lc
            self.instrument_indexes = data.instrument_indexes_lc
            self.lm_boolean = data.lm_lc_boolean
            self.nlm_boolean = data.nlm_lc_boolean
            self.lm_arguments = data.lm_lc_arguments
            self.lm_n = {}
            self.theta_iname = {}
            self.pl = pl
            self.pu = pu
            self.Ar = (self.pu - self.pl) / (2. + self.pl + self.pu)
            self.global_model = data.global_lc_model
            self.dictionary = data.lc_options
            self.numbering = data.numbering_transiting_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            # If limb-darkening, dilution factors or eclipse depth will be shared by different instruments, set the correct variable name for each:
            self.ld_iname = {}
            self.sigmaw_iname = {}
            self.mdilution_iname = {}
            self.mflux_iname = {}
            self.fp_iname = {}
            self.aglambert_iname = {}
            self.kelphomo_iname = {}
            self.kelpinhomo_iname = {}
            self.kelpthm_iname = {}
            self.phaseoffset_iname = {}
            # To make transit depth will be shared by different instruments, set the correct variable name for each:
            self.p_iname = {}
            self.p1_iname = {}
            # Since p, p1 (p2) and fp are all planetary and instrumental parameters,
            # we want to make sure that we have correct variable name for each instruments (when the instruments are shared) for _every_ planets.
            for i in self.numbering:
                self.p_iname['p' + str(i)] = {}
                self.p1_iname['p' + str(i)] = {}
                self.fp_iname['p' + str(i)] = {}
                self.aglambert_iname['p' + str(i)] = {}
                self.kelphomo_iname['p' + str(i)] = {}
                self.kelpinhomo_iname['p' + str(i)] = {}
                self.kelpthm_iname['p' + str(i)] = {}
                self.phaseoffset_iname['p' + str(i)] = {}
            self.ndatapoints_all_instruments = 0
            # Variable that turns to false only if there are no TTVs. Otherwise, always positive:
            self.Tflag = False
            # Variable that sets the total number of transit times in the whole dataset:
            self.N_TTVs = {}
            # Variable that sets if the T-parametrization will be True:
            self.Tparametrization = {}
            for pi in self.numbering:
                self.N_TTVs[pi] = 0.

            for instrument in self.inames:
                for pi in self.numbering:
                    if self.dictionary[instrument]['TTVs'][pi]['status']:
                        if self.dictionary[instrument]['TTVs'][pi][
                                'parametrization'] == 'T':
                            self.Tparametrization[pi] = True
                            self.Tflag = True
                        self.N_TTVs[pi] += self.dictionary[instrument]['TTVs'][
                            pi]['totalTTVtransits']
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(
                    self.instrument_indexes[instrument])
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[
                    instrument]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:
                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[
                        1]

                # First, check some edge cases of user input error. First, if user decided to use a_p1 and rho, raise an error:
                if ('a_p1' in self.priors.keys()) and ('rho' in self.priors.keys()):

                    raise Exception('Priors currently define a_p1 (a/Rstar) and rho (stellar density) --- these are redundant. Please choose to fit either a_p1 or rho in your fit.')

                # Now proceed with instrument namings:
                for pname in self.priors.keys():

                    # Check if variable name is a limb-darkening coefficient:
                    if pname[0:2] == 'q1' or pname[0:2] == 'u1' or pname[0:2] == 'c1':

                        vec = pname.split('_')
                        if len(vec) > 2:

                            if instrument in vec:

                                self.ld_iname[instrument] = '_'.join(vec[1:])

                        else:

                            if instrument in vec:

                                self.ld_iname[instrument] = vec[1]

                    # Check if it is a theta LM:
                    if pname[0:5] == 'theta':
                        vec = pname.split('_')
                        theta_number = vec[0][5:]
                        if len(vec) > 2:
                            if instrument in vec:
                                self.theta_iname[theta_number+instrument] = '_'.join(
                                    vec[1:])
                        else:
                            if instrument in vec:
                                self.theta_iname[theta_number+instrument] = vec[1]
                    # Check if sigma_w:
                    if pname[0:7] == 'sigma_w':
                        vec = pname.split('_')
                        if len(vec) > 3:
                            if instrument in vec:
                                self.sigmaw_iname[instrument] = '_'.join(
                                    vec[2:])
                        else:
                            if instrument in vec:
                                self.sigmaw_iname[instrument] = vec[2]
                    # Check if it is a dilution factor:
                    if pname[0:9] == 'mdilution':
                        vec = pname.split('_')
                        if len(vec) > 2:
                            if instrument in vec:
                                self.mdilution_iname[instrument] = '_'.join(
                                    vec[1:])
                        else:
                            if instrument in vec:
                                self.mdilution_iname[instrument] = vec[1]
                    if pname[0:5] == 'mflux':
                        vec = pname.split('_')
                        if len(vec) > 2:
                            if instrument in vec:
                                self.mflux_iname[instrument] = '_'.join(
                                    vec[1:])
                        else:
                            if instrument in vec:
                                self.mflux_iname[instrument] = vec[1]

                    # Planetary and instrumental parameters (e.g., fp_p1, fp_p1_inst or fp_p1_inst1_inst2):
                    for prefix, length, iname_dict, label in [('fp', 2, self.fp_iname, 'fp'),
                                                              ('aglambert', 9, self.aglambert_iname, 'aglambert'),
                                                              ('singlescat', 10, self.kelphomo_iname, 'singlescat'),
                                                              ('cml11', 5, self.kelpthm_iname, 'cml11'),
                                                              ('agkelp', 6, self.kelpinhomo_iname, 'agkelp'),
                                                              ('phaseoffset', 11, self.phaseoffset_iname, 'phaseoffset'),
                                                              ('p_', 2, self.p_iname, 'p'),
                                                              ('p1', 2, self.p1_iname, 'p1/p2')]:

                        if pname[0:length] == prefix:

                            vec = pname.split('_')
                            if len(vec) > 3:

                                # This is the case in which multiple instruments share the parameter, e.g., fp_p1_TESS1_TESS2
                                if instrument in vec:

                                    iname_dict[vec[1]][instrument] = '_' + '_'.join(vec[2:])

                            elif len(vec) == 3:

                                # This is the case of a single instrument, e.g., fp_p1_TESS
                                if instrument in vec:

                                    iname_dict[vec[1]][instrument] = '_' + vec[2]

                            elif len(vec) == 2:

                                # This adds back-compatibility so users can define a common parameter for all instruments (e.g., fp_p1):
                                iname_dict[vec[1]][instrument] = ''

                            else:

                                raise Exception('Prior for ' + label + ' is not properly defined: must be, e.g., ' + label.split('/')[0] + '_p1, ' +
                                                label.split('/')[0] + '_p1_inst or ' + label.split('/')[0] + '_p1_inst1_inst2. Currently is ' + pname)

            # Flags of whether phase curves are fit for each instrument:
            self.phase_curve = {}
            for instrument in self.inames:
                self.phase_curve[instrument] = any(self.dictionary[instrument][k] for k in
                                                   ['PhaseCurveFit', 'CowanAgolPCFit', 'LambertPCFit', 'KelpHomoPCFit',
                                                    'KelpThmPCFit', 'KelpInhomoPCFit'])

            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_lc_model

        elif modeltype == 'rv':
            self.modeltype = 'rv'
            # Inhert times, RVs, errors, indexes, etc. from data:
            self.t = data.t_rv
            self.y = data.y_rv
            self.yerr = data.yerr_rv
            self.times = data.times_rv
            self.data = data.data_rv
            self.errors = data.errors_rv
            self.instruments = data.instruments_rv
            self.ninstruments = data.ninstruments_rv
            self.inames = data.inames_rv
            self.instrument_indexes = data.instrument_indexes_rv
            self.lm_boolean = data.lm_rv_boolean
            self.nlm_boolean = data.nlm_rv_boolean
            self.lm_arguments = data.lm_rv_arguments
            self.lm_n = {}
            self.theta_iname = {}
            self.global_model = data.global_rv_model
            self.dictionary = data.rv_options
            self.numbering = data.numbering_rv_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            self.ndatapoints_all_instruments = 0
            # Go around each instrument:
            for instrument in self.inames:
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(
                    self.instrument_indexes[instrument])
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[
                    instrument]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:

                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[1]

                # Now proceed with instrument namings:
                for pname in self.priors.keys():

                    # Check if it is a theta LM:
                    if pname[0:5] == 'theta':
                        vec = pname.split('_')
                        theta_number = vec[0][5:]
                        if len(vec) > 2:
                            if instrument in vec:
                                self.theta_iname[theta_number+instrument] = '_'.join(
                                    vec[1:])
                        else:
                            if instrument in vec:
                                self.theta_iname[theta_number+instrument] = vec[1]

            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_rv_model
        else:

            raise Exception(
                'Model type "' + modeltype +
                '" not recognized. Currently it can only be "lc" for a light-curve model or "rv" for radial-velocity model.'
            )

        if self.global_model:
            self.model['global'] = np.zeros(len(self.t))
            self.model['global_variances'] = np.zeros(len(self.t))

        # Copy of the instrument names (all of them; used for the T-parametrization of TTVs):
        self.all_inames = list(self.inames)

        # Non-linear functions as functions of the parameter dictionary (functions not written with jax.numpy are
        # evaluated on the host via callbacks; see jaxmodels.jax_compatible):
        self.nlm_functions = {}
        self.uses_callbacks = False
        for instrument in self.inames:
            if self.nlm_boolean[instrument]:
                nlf = self.non_linear_functions[instrument]
                fn, is_callback = jm.jax_compatible(lambda pv, f=nlf['function'], x=nlf['regressor']: f(x, pv),
                                                    jm.example_parameter_values(self.priors),
                                                    'the non-linear function of instrument ' + instrument)
                self.nlm_functions[instrument] = fn
                self.uses_callbacks = self.uses_callbacks or is_callback

        # JAX copies of the data arrays:
        self.jy = jnp.asarray(self.y, dtype=float)
        self.jtimes = {k: jnp.asarray(self.times[k], dtype=float) for k in self.inames}
        self.jdata = {k: jnp.asarray(self.data[k], dtype=float) for k in self.inames}
        self.jerrors = {k: jnp.asarray(self.errors[k], dtype=float) for k in self.inames}
        self.jlm_arguments = {k: jnp.asarray(self.lm_arguments[k], dtype=float) for k in self.inames if self.lm_boolean[k]}

        self._check_jax_support()

        # Number of model evaluations to vectorize at once (e.g., over posterior samples or live points), chosen so that
        # each batch uses ~1/5 of the accelerator's memory (~1.2 GB on CPUs): roughly 64 floats per model point (times the
        # supersampling factor) plus N^2 for dense GPs.
        floats = 0.
        for instrument in self.inames:
            d = self.dictionary[instrument]
            n = self.ndatapoints_per_instrument[instrument]
            floats += 64. * n * (d.get('nresampling', 1) if d.get('resampling', False) else 1)
            if d.get('GPDetrend', False) and not d['noise_model'].use_celerite:
                floats += 3. * n**2
        if self.global_model and self.dictionary['global_model']['GPDetrend'] and not self.dictionary['global_model']['noise_model'].use_celerite:
            floats += 3. * len(self.t)**2
        budget_floats = 1.5e8
        try:
            memory = jax.devices()[0].memory_stats()
            if memory is not None and 'bytes_limit' in memory:
                budget_floats = max(budget_floats, 0.2 * memory['bytes_limit'] / 8.)
        except Exception:
            pass
        self.batch_size = int(np.clip(budget_floats / max(floats, 1.), 1, 4096))

        # jit-compiled functions:
        self._log_likelihood_jit = jax.jit(self.log_likelihood_fn)
        self._generate_jit = jax.jit(self._generate_fn)


class gaussian_process(object):
    """
    Given a juliet data object (created via juliet.load), a model type (i.e., is this a GP for a RV or lightcurve dataset) and
    an instrument name, this object generates a Gaussian Process (GP) object to use within the juliet library. Example usage:

               >>> GPmodel = juliet.gaussian_process(data, model_type = 'lc', instrument = 'TESS')

    celerite kernels are evaluated with celerite2.jax; the multi-dimensional squared-exponential, Matern 3/2 and exp-sine-squared
    kernels (previously evaluated with george) are evaluated with dense JAX linear algebra.

    :param data (juliet.load object)
        Object containing all the information about the current dataset. This will help in determining the type of kernel
        the input instrument has and also if the instrument has any errors associated with it to initialize the kernel.

    :param model_type: (string)
        A string defining the type of data the GP will be modelling. Can be either ``lc`` (for photometry) or ``rv`` (for radial-velocities).

    :param instrument: (string)
        A string indicating the name of the instrument the GP is being applied to. This string simplifies cross-talk with juliet's ``posteriors``
        dictionary.

    :param george_hodlr: (optional, boolean)
        Kept for back-compatibility; it has no effect.

    :param matern_eps: (optional, float)
        Epsilon parameter for the (approximate) Matern kernels.

    """

    def __new__(cls, data, *args, **kwargs):
        # Data loaded with backend = 'legacy' are handled by the legacy implementation:
        if getattr(data, 'backend', 'jax') == 'legacy':
            from .legacy.fit import gaussian_process as legacy_class
            return legacy_class(data, *args, **kwargs)
        return super().__new__(cls)

    def get_kernel_name(self, priors):

        # First, check all the GP variables in the priors file that are of the form GP_variable_instrument1_instrument2_...:
        variables_that_match = []
        for pname in priors.keys():
            vec = pname.split('_')
            if (vec[0] == 'GP') and (self.instrument in vec):
                variables_that_match = variables_that_match + [vec[1]]
        # Now we have all the variables that match the current instrument in variables_that_match. Check which of the
        # implemented GP models gives a perfect match to all the variables; that will give us the name of the kernel:
        n_variables_that_match = len(variables_that_match)
        if n_variables_that_match == 0:
            raise Exception(
                'Input error: it seems instrument ' + self.instrument +
                ' has no defined priors in the prior file for a Gaussian Process. Check the prior file and try again.'
            )

        for kernel_name in self.all_kernel_variables.keys():
            if sorted(self.all_kernel_variables[kernel_name]) == sorted(variables_that_match):
                return kernel_name

        raise Exception('Input error: GP hyperparameters ' + ', '.join(variables_that_match) + ' for instrument ' +
                        self.instrument + ' do not match any of the implemented kernels.')

    def set_input_instrument(self, input_variables):

        # This function sets the "input instrument" (self.input_instrument) name for each variable (self.variables).
        # If, for example, GP_Prot_TESS_K2_rv and GP_Gamma_TESS, and self.variables = ['Prot','Gamma'],
        # then self.input_instrument = ['TESS_K2_rv','TESS'].
        self.input_instrument = []
        self.parameter_names = {}
        for GPvariable in self.variables:
            for pname in input_variables.keys():
                vec = pname.split('_')
                if (vec[0] == 'GP') and (vec[1] == GPvariable) and (self.instrument in vec):
                    self.input_instrument.append('_'.join(vec[2:]))
                    self.parameter_names[GPvariable] = pname

    def hyperparameters(self, parameter_values):
        """Dictionary with the values of the hyperparameters of the kernel."""
        return {v: parameter_values[self.parameter_names[v]] for v in self.variables}

    def log_likelihood(self, parameter_values, residuals, variances):
        """
        GP log-likelihood of the residuals. ``variances`` are the variances added to the diagonal of the covariance
        matrix (errorbars squared plus jitter terms squared).
        """
        return self.GP.log_likelihood(self.hyperparameters(parameter_values), residuals, variances)

    def predict(self, parameter_values, residuals, variances, X=None):
        """Mean GP prediction at regressors X (default: the regressors of the fit) conditioned on the residuals."""
        return self.GP.predict(self.hyperparameters(parameter_values), residuals, variances, X)

    def __init__(self,
                 data,
                 model_type,
                 instrument,
                 george_hodlr=True,
                 matern_eps=0.01):
        self.isInit = False
        self.model_type = model_type.lower()
        # Perform changes that define the model_type. For example, the juliet input sigmas (both jitters and GP amplitudes) are
        # given in ppm in the input files, whereas for RVs they have the same units as the input RVs. This conversion factor is
        # defined by the model_type:
        if self.model_type == 'lc':
            if instrument is None:
                instrument = 'lc'
            self.sigma_factor = 1e-6
        elif self.model_type == 'rv':
            if instrument is None:
                instrument = 'rv'
            self.sigma_factor = 1.
        else:
            raise Exception(
                'Model type ' + model_type +
                ' currently not supported. Only "lc" or "rv" can serve as inputs for now.'
            )

        # Name of input instrument if given:
        self.instrument = instrument

        # Initialize global model variable:
        self.global_GP = False

        # Extract information from the data object:
        if self.model_type == 'lc':
            # Save input predictor:
            if instrument == 'lc':
                self.X = data.GP_lc_arguments['lc']
                self.global_GP = True
            else:
                self.X = data.GP_lc_arguments[instrument]
            # Save errors (if any):
            if data.yerr_lc is not None:
                if instrument != 'lc':
                    self.yerr = data.yerr_lc[
                        data.instrument_indexes_lc[instrument]]
                else:
                    self.yerr = data.yerr_lc
            else:
                self.yerr = None
        elif self.model_type == 'rv':
            # Save input predictor:
            if instrument == 'rv':
                self.X = data.GP_rv_arguments['rv']
                self.global_GP = True
            else:
                self.X = data.GP_rv_arguments[instrument]
            # Save errors (if any):
            if data.yerr_rv is not None:
                if instrument != 'rv':
                    self.yerr = data.yerr_rv[
                        data.instrument_indexes_rv[instrument]]
                else:
                    self.yerr = data.yerr_rv
            else:
                self.yerr = None

        # Fix sizes of regressors if wrong:
        self.X = np.asarray(self.X, dtype=float)
        if len(self.X.shape) == 2:
            if self.X.shape[1] != 1:
                self.nX = self.X.shape[1]
            else:
                self.X = self.X[:, 0]
                self.nX = 1
        else:
            self.nX = 1

        # Define all possible kernels available by the object:
        self.all_kernel_variables = kernel_variables(self.nX)

        # Find kernel name (and save it to self.kernel_name):
        self.kernel_name = self.get_kernel_name(data.priors)
        self.variables = self.all_kernel_variables[self.kernel_name]
        self.use_celerite = 'Celerite' in self.kernel_name
        self.george_hodlr = george_hodlr
        self.set_input_instrument(data.priors)

        # Regressors of celerite kernels don't need to be sorted (this is handled internally). However, as in previous juliet
        # versions, the data of global models are sorted by the GP regressor (this is done by juliet.load when isInit is False):
        if self.use_celerite and self.global_GP:
            if not (np.all(np.diff(self.X) >= 0) or np.all(np.diff(self.X) <= 0)):
                return

        # Initialize the (JAX) GP object:
        self.legacy_gp_parametrization = getattr(data, 'legacy_gp_parametrization', False)
        self.GP = JaxGP(self.kernel_name, self.X, self.sigma_factor, matern_eps=matern_eps,
                        legacy_parametrization=self.legacy_gp_parametrization)
        self.isInit = True
