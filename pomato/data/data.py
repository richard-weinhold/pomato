# pylint: disable-msg=E1101

import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pomato
import pomato.tools as tools
from pomato.data.results import Results
from pomato.data.worker import DataWorker


class Timeseries(): 
    """Timeseries returns Day-Ahead or Real-Time timeseries depending n input data and settings."""

    def __init__(self):
        self.data = None 
        self.includes_da_demand = False
        self.includes_da_availability = False
        
        self._demand_el_da = pd.DataFrame()
        self._demand_el_rt = pd.DataFrame()
        self._availability_da = pd.DataFrame()
        self._availability_rt = pd.DataFrame()

    def load_data(self, data):
        self.data = data

        if "demand_el_da" in data.data_attributes:  
            self._demand_el_da = self.data.demand_el_da
            self._demand_el_rt = self.data.demand_el_rt
            self.includes_da_demand = True
            self.data.model_structure["demand_el"].update(
                {"demand_el_da": {'type': 'float', 'default': np.nan}})

        else:
            self.includes_da_demand = False

        if "availability_da" in data.data_attributes:
            self._availability_da = self.data.availability_da
            self._availability_rt = self.data.availability_rt
            self.includes_da_availability = True
            self.data.model_structure["availability"].update(
                {"availability_da": {'type': 'float', 'default': np.nan}})
        else:
            self.includes_da_availability = False

    @property
    def demand_el(self):
        if self.includes_da_demand:
            return pd.merge(self._demand_el_rt, self._demand_el_da, 
                            on=["timestep", "node"], suffixes=["", "_da"])
        else:
            return self._demand_el_rt
    
    @property
    def availability(self):
        if self.includes_da_availability:
            return pd.merge(self._availability_rt, self._availability_da, 
                            on=["timestep", "plant"], suffixes=["", "_da"])
        else:
            return self._availability_rt


class DataManagement():
    """The DataManagement class provides processed data to all other modules in POMATO

    This is done by managing the data read-in, processing and validation of
    input data and facilitation of results.

    Parameters
    ----------
    wdir : pathlib.Path
       Working directory, proliferating from POMATO main module.
    options : dict
        The options from POMATO main method persist in the DataManagement.

    Attributes
    ----------
    wdir : pathlib.Path
        Working directory
    options : dict
        The options from POMATO main method persist in the DataManagement.
    data_structure : dict
        data_structure contains the structure of the input data including the
        name i.e. table like nodes and its attributes e.g. zones, lat/lon etc
    model_structure : dict
        Similarity ``data_structure``, model_structure defines the data necessary
        to run the market model and its attributes. This is used to verify the
        processed input data in this module.
    data_attributes : dict
        All input and model data is available as an attribute of this method.
        ``data_attributes`` is used to initialize them and to track if this data
        is loaded from file as indicated by the dict{attr, bool}.
    results : dict(str, :obj:`~pomato.data.Results`)
        The Results module allows to attach and process results from the market model
        alongside the DataManagement module. This ensures consitency of the data
        and results. The dict results can store multiple results.
    """

    def __init__(self, options, wdir):
        # import logger
        self.logger = logging.getLogger('Log.MarketModel.DataManagement')
        self.logger.info("Initializing DataObject")

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])
        self.options = options

        self.data_structure = {}
        self.model_structure = self.load_model_structure()
        # self.load_structure()
        self.missing_data = []
        self.data_validation_report = {}
        self.model_validation_report = {}

        self.data_attributes = {data: False for data in list(self.model_structure)}
        self.data_source = None
        self.includes_da_data = False  
        self.includes_da_availability = False      
        # All data as Attributes of DataManagement Module
        for attr in self.data_attributes:
            if not attr in ["availability", "demand_el"]:
                cols = [col for col in self.model_structure[attr].keys() if col != "index"]
                setattr(self, attr, pd.DataFrame(columns=cols))
        
        self.timeseries = Timeseries()
        # Results are part of the results processing, but attached to the data module.
        self.results = {}

    @property
    def demand_el(self):
        return self.timeseries.demand_el

    @demand_el.setter
    def demand_el(self, demand_el):
        self.timeseries._demand_el_rt = demand_el[["timestep", "node", "demand_el"]]
        # self.timeseries._demand_el_rt = demand_el
    
    @property
    def availability(self):
        return self.timeseries.availability

    @availability.setter
    def availability(self, availability):
        # self.timeseries._availability_rt = availability
        self.timeseries._availability_rt = availability[["timestep", "plant", "availability"]]

    def save_data(self, filepath):
        """Write Data to excel file.

        This methods provides a way to save a working data set, which then can be imported without
        processing.

        Parameters
        ----------
         filepath: pathlib.Path
            Filepath to an .xls or .xlsx file, should include file extension.
        """
        self.logger.info("Writing Data to Excel File %s", str(filepath))
        with pd.ExcelWriter(filepath.with_suffix(".xlsx")) as writer:  # pylint: disable=abstract-class-instantiated
            for data in self.data_attributes:
                if len(getattr(self, data)) > 1048576:
                    self.logger.info("Flattening %s", data)
                    cols = getattr(self, data).columns
                    getattr(self, data).pivot(
                        index=cols[0], columns=cols[1], values=cols[2]).to_excel(writer, sheet_name=data)
                else:
                    getattr(self, data).to_excel(writer, sheet_name=data)

        self.logger.info("Writing Data to an archive of csv files %s", str(filepath))
        if not filepath.is_dir():
            filepath.mkdir()
        for data in self.data_attributes:
            getattr(self, data).to_csv(filepath.joinpath(data + ".csv"))
        self.logger.info("zipping....")
        shutil.make_archive(filepath, 'zip', filepath)
        shutil.rmtree(filepath, ignore_errors=True)
        self.logger.info("saved!")
    
    def save_results(self, folder, name=None):
        """Copy the loaded results into a folder.

        Parameters
        ----------
         folder: pathlib.Path
            Folder where all results are copied to.
        """
        if len(self.results) == 0:
            self.logger.warning("No results to save")
        else:
            for key in self.results:
                if name:
                    if any(char.isdigit() for char in key):
                        result_folder = name + key[9:]
                    else:
                        result_folder = name + "_" + key
                else:
                    result_folder = key
                folder.joinpath(result_folder).mkdir()
                tools.copytree(self.results[key].result_attributes["source"], folder.joinpath(result_folder))

    def load_data(self, filepath):
        """Load Data from dataset at filepath.

        Currently .xls(x) and .mat work and filepath with or without ext.
        After the raw data is read it is processed in the process_input
        method of this module.

        Parameters
        ----------
         filepath: pathlib.Path
            Filepath .mat or .xlsx file. There are some path arithmetics to
            catch errors but not a lot.
        """
        # Make sure wdir/file_path or wdir/data/file_path is a file
        if self.wdir.joinpath(filepath).is_file():
            DataWorker(self, self.wdir.joinpath(filepath))
        elif self.wdir.joinpath(f"data_input/{filepath}").is_file():
            DataWorker(self, self.wdir.joinpath(f"data/{filepath}"))
        elif self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat").is_file():
            DataWorker(self, self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat"))
        elif self.wdir.joinpath(filepath).is_dir():
            DataWorker(self,self.wdir.joinpath(filepath))
        else:
            self.logger.error("Data File not found!")
            raise FileNotFoundError
        
        if len(self.missing_data) > 0:
            self.logger.warning(("Not complete list of expected input data found. See .missing_data"))

        self.process_input()
        self.data_source = filepath

    def validate_inputdata(self):
        """Validate input data and compare it to the predefined data structure.

        This method validates the input data and checks whether it fits the
        predefined structure. This makes it easy to find errors and nan
        values in the data.

        Input data is generally defined by:
            * table/name/data: e.g. nodes, the general data type which always
              has an index.
            * attributes: e.g. zone of a node, attributes of the data.
            * type: the type can be int/float etc or the index of another
              table. E.g. zone is of type zone.index, so the index of another table.
            * optional: bool to indicate whether a nan/missing value is an
              error. E.g. the attribute name is optional, as some nodes might
              not have a specific name assigned and are solely identified
              by the index.

        Therefore, this method check for all data if its required attributes
        are available and reports missing ones in `missing_data`. Additionally
        it checks the values for each attribute to be of the defined type and
        if it refers to another data's index, checks for consistency. Invalid
        values are removed in this process and reported. This can indicate a
        problem with the input data, but also wanted (in most cases).
        """
        self.logger.info("Validating Input Data...")
        
        self.data_validation_report = {"missing_data": {}, "removed_data": {}}
        for data in self.data_structure.index.unique():  
            # Data refers to nodes, plants etc...
            # Distinguish between required and optional attributes
            attributes = self.data_structure.loc[[data], :].set_index("attributes")
            required_attr = list(attributes.index[attributes.optional & (attributes.index != "index")])
            optional_attr = list(attributes.index[~attributes.optional & (attributes.index.isin(getattr(self, data).columns))])
            condition = ((~attributes.optional) 
                         & (~attributes.index.isin(getattr(self, data).columns)) 
                         & (attributes.index != "index"))
            missing_opt_attributes = list(attributes.index[condition])
            self.data_validation_report["missing_data"][data] = {"optional": missing_opt_attributes}
            reference_attributes = attributes.loc[attributes.type.str.contains(".", regex=False), :]
            # Log error when required attribute is missing in data and store in missing_data
            # Missing means that the column of the attribute does not exist!
            if not attributes[(attributes.index.isin(required_attr) &
                            (~attributes.index.isin(getattr(self, data).columns)))].empty:
                condition = (attributes.index.isin(required_attr) &
                            (~attributes.index.isin(getattr(self, data).columns)))
                self.data_validation_report["missing_data"][data]["required"] = list(attributes.index[condition])

            else:  # No required attribute is missing, continue with checking the contents of each column
                self.data_validation_report["removed_data"][data] = {"reference": [], "reference_nans": []}
                tmp = getattr(self, data).loc[:, required_attr + optional_attr]
                for attr, ref in zip(reference_attributes.index, reference_attributes.type):
                    ref_data, ref_attr = ref.split(".")
                    if ref_attr == "index":
                        reference_keys = getattr(self, ref_data).index
                    else:
                        reference_keys = getattr(self, ref_data)[ref_attr]

                    if attr in required_attr and not tmp.loc[~(tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(tmp[attr].isin(reference_keys))]
                        self.data_validation_report["removed_data"][data]["reference"].append(attr)
                    elif not tmp.loc[(~tmp[attr].isna()) & (~tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(~tmp[attr].isna()) & (tmp[attr].isin(reference_keys))]
                        self.data_validation_report["removed_data"][data]["reference_nans"].append(attr)
                setattr(self, data, tmp.infer_objects())

        self.data_validation_report = tools.remove_empty_subdicts(self.data_validation_report)
        if len(self.data_validation_report) > 0: 
            self.logger.warning(("Data validation completed with warnings. See the .data_validation_report."))
        else:
            self.logger.warning("Data validation completed with no issues.")


    def process_input(self):
        """Input Processing to bring input data is the desired pomato format.

        This is done in a separate module :py:obj:`~pomato.data.InputProcessing`
        and should result in a dataset as defined in the model_structure
        attribute. The instance of :py:obj:`~pomato.data.InputProcessing` does not persist as it
        only processes the data one.

        The processed data gets validated based on the predefined
        input and model data structures.
        """

        self.validate_inputdata()
        self.timeseries.load_data(self)    
        self.validate_modeldata()

    def load_model_structure(self):
        """Load model structure as part of init."""
        with open(self.package_dir.joinpath("data/model_structure.json"), "r") as jsonfile:
            model_structure = json.load(jsonfile)
        
        for k in model_structure:
            for kk in model_structure[k]:
                if model_structure[k][kk]["default"] == "none":
                    model_structure[k][kk]["default"] = np.nan
        return model_structure        
        
    def validate_modeldata(self):
        """Validate the processed input data to be conform with predefined model data structure.

        Analogues to the :meth:`~validate_inputdata` method, this methods checks the consistency
        of the input data. In contrast to :meth:`~validate_inputdata` this method compares the
        processed data to the predefined model structure and adds default values or empty tables
        to allow different data sets run in POMATO.

        For example: the IEEE118 case study does not containt heat demand or generation, also are
        there no timeseries to cover adjacent regions via net export parameters. So the input data
        can cover a subjet of possible data set but the model data has to be consistent.
        """
        self.model_validation_report = {"empty": [], "default_values": {}}
        for data in self.model_structure:
            # if getattr(self, data).empty:
            if len(getattr(self, data)) == 0:
                cols = [col for col in self.model_structure[data].keys() if col != "index"]
                setattr(self, data, pd.DataFrame(columns=cols))
                self.model_validation_report["empty"].append(data)
                # self.logger.warning("%s not in Input Data, initialized as empty", data)
            else:
                self.model_validation_report["default_values"][data] = {}
                tmp = getattr(self, data)
                cols = [col for col in self.model_structure[data].keys() if col != "index"]
                
                # see if attribute is in data, if not -> add columns with default value
                for attr in [col for col in cols if col not in tmp.columns]:
                    default_value = self.model_structure[data][attr]["default"]
                    tmp.loc[:, attr] = default_value
                    self.model_validation_report["default_values"][data][attr] = default_value
                    # self.logger.warning("Attribute %s not in %s, initialized as %s", attr, data, str(default_value))
                
                # add missing values as default
                for attr in cols:
                    default_value = self.model_structure[data][attr]["default"]
                    if any(tmp[attr].isna()) and not pd.isna(default_value):
                        tmp.loc[tmp[attr].isna(), attr] = default_value
                        self.model_validation_report["default_values"][data][attr] = default_value
                        # self.logger.warning("Attribute %s in %s contains NaNs, initialized as %s", attr, data, str(default_value))
                setattr(self, data, tmp)
        

        if len(self.model_validation_report["empty"]) > 0:
            self.logger.warning(("Some data was initialized empty. See model_validation_report."))
        self.model_validation_report["default_values"] = tools.remove_empty_subdicts(self.model_validation_report["default_values"])
        if len(self.model_validation_report["default_values"]) > 0:
            self.logger.warning(("Some data missing or contained NaNs. See model_validation_report."))


    def process_results(self, result_folder, grid):
        """Initialize :class:`~pomato.data.Results` with `results_folder` and the own instance."""
        self.results[result_folder.name] = Results(self, grid, result_folder)

    def return_results(self, redispatch=True):
        """Interface method to allow access to results from :class:`~pomato.data.Results`."""

        if redispatch and len(self.results) > 1:
            redispatch_results = [r for r in list(self.results) if "redispatch" in r]
            market_result = [r for r in list(self.results) if "market_result" in r]
            if len(redispatch_results) == 1 and len(market_result) == 1:
                return self.results[market_result[0]], self.results[redispatch_results[0]]
            else:
                self.logger.error("Multiple results initialized that fit criteria")
        elif len(self.results) == 1:
            return next(iter(self.results.values()))
        else:
            self.logger.error("Multiple results initialized that fit criteria")
      
    def set_default_net_position(self, net_position):
        """Add default net position."""
        self.net_position = pd.DataFrame(index=self.demand_el.timestep.unique(), 
                                                columns=self.zones.index, 
                                                data=net_position).stack().reset_index()

        self.net_position.columns = [col for col in self.model_structure["net_position"].keys() if col != "index"]

    def process_inflows(self):
        """Process inflows to (hydro-) storages.

        If no raw data create an all zero timeseries for all electric storage (plant_type es)
        power plants
        """
        inflows_columns = [col for col in self.model_structure["inflows"].keys() if col != "index"]
        self.inflows = pd.DataFrame(columns=inflows_columns)
        self.inflows["timestep"] = self.demand_el.timestep.unique()

        tmp = self.inflows.pivot(index="timestep", columns="plant", values="inflow").fillna(0)
        condition = self.plants.plant_type.isin(self.options["optimization"]["plant_types"]["es"])
        for es_plant in self.plants.index[condition]:
            if es_plant not in tmp.columns:
                tmp[es_plant] = 0
        self.inflows = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="inflow").dropna()

    def unique_mc(self):
        """Make marginal costs unique.

        This is done by adding a small increment multiplied by the number if plants with the
        same mc. This makes the solver find a unique solution (at least in regards to generation
        schedule) and is sopposed to have positive effect on solvetime.
        """
        for marginal_cost in self.plants.mc_el:
            condition_mc = self.plants.mc_el == marginal_cost
            self.plants.loc[condition_mc, "mc"] = \
            self.plants.mc_el[condition_mc] + \
            [int(x)*1E-4 for x in range(0, len(self.plants.mc_el[condition_mc]))]

    def line_susceptance(self):
        """Calculate line susceptance for lines that have none set.

        This is not maintained as the current grid data set includes this parameter. However, this
        Was done with the simple formula b = length/type ~ where type is voltage level. While this
        is technically wrong, it works with linear load flow, as it only relies on the
        conceptual "conductance"/"resistance" of each circuit/line in relation to others.
        """
        if ("x per km" in self.lines.columns)&("voltage" in self.nodes.columns):
            self.lines['x'] = self.lines['x per km'] * self.lines["length"] * 1e-3
            self.lines.loc[self.lines.technology == "transformer", 'x'] = 0.01
            base_mva = 100
            base_kv = self.nodes.loc[self.lines.node_i, "voltage"].values
            # base_kv = 110
            v_base = base_kv * 1e3
            s_base = base_mva * 1e6
            z_base = np.power(v_base,2)/s_base
            self.lines['x'] = np.divide(self.lines['x'], z_base)
            self.lines['b'] = np.divide(1, self.lines['x'])
