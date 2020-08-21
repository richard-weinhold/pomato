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
import pomato.data.input_data_processing as process
from pomato.data.results import ResultProcessing
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
            self.data.model_structure["demand_el"].update({"demand_el_da": {'type': 'float', 'default': np.nan}})

        else:
            self.includes_da_demand = False

        if "availability_da" in data.data_attributes:
            self._availability_da = self.data.availability_da
            self._availability_rt = self.data.availability_rt
            self.includes_da_availability = True
            self.data.model_structure["availability"].update({"availability_da": {'type': 'float', 'default': np.nan}})
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
       Working directory, proliferating from POAMTOs main module.
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
    results : dict(str, :obj:`~pomato.data.ResultProcessing`)
        The ResultProcessing module allows to attach and process results from the market model
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
            Filepath to an xlsx file, should include file extension.
        """
        self.logger.info("Writing Data to Excel File %s", str(filepath))
        with pd.ExcelWriter(filepath.with_suffix(".xlsx")) as writer: #pylint: disable=abstract-class-instantiated
            for data in self.data_attributes:
                if len(getattr(self, data)) > 1048576:
                    self.logger.info("Flattening %s", data)
                    cols = getattr(self, data).columns
                    getattr(self, data).pivot(index=cols[0], columns=cols[1], values=cols[2]).to_excel(writer, sheet_name=data)
                else:
                    getattr(self, data).to_excel(writer, sheet_name=data)

        self.logger.info("Writing Data to an archive of csv's %s", str(filepath))
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
        ### Make sure wdir/file_path or wdir/data/file_path is a file
        if self.wdir.joinpath(filepath).is_file():
            DataWorker(self, self.wdir.joinpath(filepath))

        elif self.wdir.joinpath(f"data_input/{filepath}").is_file():
            DataWorker(self, self.wdir.joinpath(f"data/{filepath}"))

        elif self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat").is_file():
            DataWorker(self, self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat"))
        else:
            self.logger.error("Data File not found!")
            raise FileNotFoundError

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
        self.missing_data = {}
        for data in self.data_structure.index.unique():  # Data refers to nodes, plants etc...
            # Distinguish between required and optional attributes
            attributes = self.data_structure.loc[[data], :].set_index("attributes")
            required_attr = list(attributes.index[attributes.optional & (attributes.index != "index")])
            optional_attr = list(attributes.index[~attributes.optional & (attributes.index.isin(getattr(self, data).columns))])
            missing_opt_attributes = list(attributes.index[~attributes.optional & 
                                               (~attributes.index.isin(getattr(self, data).columns))])

            self.missing_data[data] = {"optional": missing_opt_attributes}
        
            reference_attributes = attributes.loc[attributes.type.str.contains(".", regex=False), :]
            # Log error when required attribute is missing in data and store in missing_data
            # Missing means that the column of the attribute does not exist!
            if not attributes[(attributes.index.isin(required_attr) &
                            (~attributes.index.isin(getattr(self, data).columns)))].empty:
                condition = (attributes.index.isin(required_attr) &
                            (~attributes.index.isin(getattr(self, data).columns)))
                self.missing_data[data]["required"] = list(attributes.index[condition])
                self.logger.warning("Required Data not there as expected in %s", data)

            else:  # No required attribute is missing, continue with checking the contents of each column
                tmp = getattr(self, data).loc[:, required_attr + optional_attr]
                for attr, ref in zip(reference_attributes.index, reference_attributes.type):
                    ref_data, ref_attr = ref.split(".")
                    if ref_attr == "index":
                        reference_keys = getattr(self, ref_data).index
                    else:
                        reference_keys = getattr(self, ref_data)[ref_attr]
                    if attr in required_attr and not tmp.loc[~(tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(tmp[attr].isin(reference_keys))]
                        self.logger.warning("Invalid Reference Keys and NaNs removed for %s in %s", attr, data)
                    elif not tmp.loc[(~tmp[attr].isna()) & (~tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(~tmp[attr].isna()) & (tmp[attr].isin(reference_keys))]
                        self.logger.warning("Invalid Reference Keys without NaNs removed for %s in %s", attr, data)
                setattr(self, data, tmp.infer_objects())
            
            self.missing_data = tools.remove_empty_subdicts(self.missing_data)
            for k in self.missing_data:
                if "required" in self.missing_data[k].keys():
                    self.logger.error("attributes missing in %s", k)


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

    def default_net_position(self, net_position):
        process.set_default_net_position(self, net_position)

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
        for data in self.model_structure:
            # if getattr(self, data).empty:
            if len(getattr(self, data)) == 0:
                cols = [col for col in self.model_structure[data].keys() if col != "index"]
                setattr(self, data, pd.DataFrame(columns=cols))
                self.logger.warning("%s not in Input Data, initialized as empty", data)
            else:
                tmp = getattr(self, data)
                cols = [col for col in self.model_structure[data].keys() if col != "index"]
                
                # see if attribute is in data -> add columns with default value
                for attr in [col for col in cols if col not in tmp.columns]:
                    default_value = self.model_structure[data][attr]["default"]
                    tmp.loc[:, attr] = default_value
                    self.logger.warning("Attribute %s not in %s, initialized as %s", attr, data, str(default_value))
                
                # add missing values as default
                for attr in cols:
                    default_value = self.model_structure[data][attr]["default"]
                    if any(tmp[attr].isna()) and not pd.isna(default_value):
                        tmp.loc[tmp[attr].isna(), attr] = default_value
                        self.logger.warning("Attribute %s in %s contains NaNs, initialized as %s", attr, data, str(default_value))

                setattr(self, data, tmp)

    def process_results(self, result_folder, grid):
        """Initialize :class:`~pomato.data.ResultProcessing` with `results_folder` and the own instance."""
        self.results[result_folder.name] = ResultProcessing(self, grid, result_folder)

    def return_results(self, redispatch=True):
        """Interface method to allow access to results from :class:`~pomato.data.ResultProcessing`."""

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
      
    def _clear_all_data(self):
        attr = list(self.__dict__.keys())
        attr.remove('logger')
        for att in attr:
            delattr(self, att)

    def visualize_inputdata(self, folder=None, show_plot=True):
        """Create default Plots for Input Data.

        This methods is currently not maintained, but was thought to provide standard figures
        to visualize the (processed) input data.
        """
        if folder and not Path.is_dir(folder):
            self.logger.warning("Folder %s does not exist!", folder)
            self.logger.warning("Creating %s", folder)
            Path.mkdir(folder)
        
        if show_plot or not folder:
            plt.ion()
        else:
            plt.ioff()
        # Demand by Zone
        demand_zonal = pd.DataFrame(index=self.demand_el.timestep.unique())

        for zone in self.zones.index:
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
            cond_node_in_zone = self.demand_el.node.isin(nodes_in_zone)
            demand_zonal[zone] = self.demand_el[cond_node_in_zone].groupby("timestep").sum()
            
        fig_demand, ax_demand = plt.subplots()
        demand_zonal.plot.area(ax=ax_demand, xticks=np.arange(0, len(demand_zonal.index), 
                               step=len(demand_zonal.index)/10))
        ax_demand.legend(loc='upper right')
        ax_demand.margins(x=0)

        # Plot Installed Capacity by....
        plants_zone = pd.merge(self.plants, self.nodes.zone, how="left", 
                            left_on="node", right_index=True)

        inst_capacity = (plants_zone[["g_max", "zone", "plant_type"]]
                         .groupby(["plant_type", "zone"], as_index=False).sum())
        fig_gen, ax_gen = plt.subplots()
        inst_capacity.pivot(index="zone", columns="plant_type",
                            values="g_max").plot.bar(stacked=True, ax=ax_gen)

        ax_gen.legend(loc='upper right')
        ax_gen.margins(x=0)
        
        if folder:
            fig_demand.savefig(str(folder.joinpath("zonal_demand.png")))
            fig_gen.savefig(str(folder.joinpath(f"installed_capacity_by_type.png")))
