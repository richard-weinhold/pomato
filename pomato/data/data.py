# pylint: disable-msg=E1101

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pomato.data.input import InputProcessing
from pomato.data.results import ResultProcessing
from pomato.data.worker import DataWorker


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
        Similarily ``data_structure``, model_structure defines the data nessesary
        to run the market model and its attributes. This is used to varify the
        processed input data in this module.
    data_attributes : dict
        All input and model data is available as an attribute of this method.
        ``data_attributes`` is used to initialize them and to track if this data
        is loaded from file as indicated by the dict{attr, bool}.
    results : dict(str, :obj:`~pomato.data.ResultProcessing`)
        The ResultProcessing module allows to attach and process results from the market model
        alongside the DataManagement module. This ensures consitency of the data
        and results. The dict reuslts can store mutiple results.
    """

    def __init__(self, options, wdir):
        # import logger
        self.logger = logging.getLogger('Log.MarketModel.DataManagement')
        self.logger.info("Initializing DataObject")

        self.wdir = wdir
        self.options = options

        self.data_structure = {}
        self.model_structure = {}
        self.load_structure()

        self.data_attributes = {data: False for data in set(list(self.data_structure)
                                                            + list(self.model_structure))}

        self.data_source = None
        # All data as Attributes of DataManagement Module
        for attr in self.data_attributes:
            setattr(self, attr, pd.DataFrame())
        # Results are part of the results processing, but attached to the data module.
        self.results = {}

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
        with pd.ExcelWriter(filepath) as writer:
            for data in self.data_attributes["data"]:
                getattr(self, data).to_excel(writer, sheet_name=data)

    def load_data(self, filepath):
        """Load Data from dataset at filepath.

        Currently .xls(x) and .mat work and filepath with or without ext.
        After the raw data is read it is processed in the process_input
        method of this module.

        Parameters
        ----------
         filepath: pathlib.Path
            Filepath .mat or .xlsx file. There are some path arithmetics to
            catch errors but not alot.
        """
        ### PATH ARETMETICS INCOMING
        ### Make sure wdir/file_path or wdir/data/file_path is a file
        if self.wdir.joinpath(filepath).is_file():
            DataWorker(self, self.wdir.joinpath(filepath))
            self.process_input()
            self.data_source = filepath

        elif self.wdir.joinpath(f"data_input/{filepath}").is_file():
            DataWorker(self, self.wdir.joinpath(f"data/{filepath}"))
            self.process_input()
            self.data_source = filepath

        elif self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat").is_file():
            DataWorker(self, self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat"))
            self.process_input()
            self.data_source = filepath
        else:
            self.logger.error("Data File not found!")
            raise FileNotFoundError

    def stack_data(self):
        """Stacks data that comes in a wide format.

        As xls files have a limit when it comes to table length, some data will
        be stored wide instead (e.g. demand or availability). To properly use
        this data in the model the data will be stacked based on the
        configuration in the options file (options["data"]["stacked"]).

        """
        for data in self.options["data"]["stacked"]:
            tmp = getattr(self, data)
            tmp = tmp.stack().reset_index()
            tmp.columns = self.data_structure[data]["attributes"][1:]
            setattr(self, data, tmp.infer_objects())

    def validate_inputdata(self):
        """Validate input data and compare it to the predefined data structure.

        This method validatesy the input data and checks whether it fits the
        predefined structeure. This makes it easy to find errors and nan
        values in the data.

        Input data is generally defined by:
            * table/name/data: e.g. nodes, the general data type which always
              has an index.
            * attributes: e.g. zone of a node, atrributes of the data.
            * type: the type can be int/float etc or the index of another
              table. E.g. zone is of type zone.index, so the index of another table.
            * optional: bool to indicate whether a nan/missing value is an
              error. E.g. the attribute name is optional, as some nodes might
              not have a specific name assigned and are solely iodentified
              by the index.

        Therfore, this method check for all data if its required attributes
        are available and reports missing ones in `missing_data`. Additionally
        it checks the values for each attribute to be of the defined type and
        if it refers to another data's index, cheks for consistency. Invalid
        values are removed in this process and reported. This can indicate a
        problem with the input data, but also wanted (in most cases).
        """
        self.logger.info("Validating Input Data...")
        self.missing_data = {}
        for data in self.data_structure:  # Data refers to nodes, plants etc...
            # Distinguish between required and optional attributes
            attributes = self.data_structure[data]
            required_attr = [attr for attr in attributes.loc[(attributes["attributes"] != "index") &
                                                             (~attributes["optional"].astype(bool)),
                                                             "attributes"]]
            optional_attr = [attr for attr in attributes.loc[(attributes["attributes"] != "index") &
                                                             (attributes["optional"].astype(bool)),
                                                             "attributes"] if attr in getattr(self, data).columns]

            ref_attr = attributes.loc[attributes.type.str.contains(".", regex=False)]

            # Log error when required attribute is missing in data and store in missing_data
            # Missing means that the column of the attribute does not exist!
            if not attributes[(attributes.attributes.isin(required_attr) &
                               (~attributes.attributes.isin(getattr(self, data).columns)))].empty:
                condition = (attributes.attributes.isin(required_attr) &
                             (~attributes.attributes.isin(getattr(self, data).columns)))
                self.missing_data[data] = list(attributes.attributes[condition])
                self.logger.error("Required Data not there as expexted in %s", data)

            else:  # No required attribute is missing, continue with checking the contents of each column
                tmp = getattr(self, data).loc[:, required_attr + optional_attr]
                for attr, ref in zip(ref_attr.attributes, ref_attr.type):
                    ref_data, ref_attr = ref.split(".")
                    if ref_attr == "index":
                        reference_keys = getattr(self, ref_data).index
                    else:
                        reference_keys = getattr(self, ref_data)[ref_attr]
                    if attr in required_attr and not tmp.loc[~(tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(tmp[attr].isin(reference_keys))]
                        self.logger.error("Invalid Reference Keys and NaNs removed for %s in %s", attr, data)
                    elif not tmp.loc[(~tmp[attr].isna()) & (~tmp[attr].isin(reference_keys))].empty:
                        tmp = tmp.loc[(~tmp[attr].isna()) & (tmp[attr].isin(reference_keys))]
                        self.logger.error("Invalid Reference Keys without NaNs removed for %s in %s", attr, data)
                setattr(self, data, tmp.infer_objects())

        if self.missing_data:
            for key in self.missing_data:
                self.logger.error("attributes missing in %s: %s", key, " ".join(self.missing_data[key]))

    def process_input(self):
        """Input Processing to bring input data is the desired pomato format.

        This is done in a seperate module :py:obj:`~pomato.data.InputProcessing`
        and should result in a dataset as defined in the model_structure
        attribute. The instance of :py:obj:`~pomato.data.InputProcessing` does not persist as it
        only processes the data one.

        The processed data gets validated based on the predifined
        input and model data structures.
        """
        if "stacked" in self.options["data"]:
            self.stack_data()
        if self.options["data"]["process_input"]:
            InputProcessing(self)
        else:
            self.logger.info("Input Data not processed")

        self.validate_inputdata()
        self.validate_modeldata()

    def load_structure(self):
        """Init Model- and Data structure from file based on option `data type`."""
        file = self.wdir.joinpath("data_input/data_structure.xlsx")
        xls = pd.ExcelFile(file)
        structure = xls.parse(self.options["data"]["data_type"])
        columns = [c for c in structure.columns if "Unnamed:" not in c]
        self.data_structure = {}
        for c in columns:
            col_pos = structure.columns.get_loc(c)
            cols = list(structure.columns[col_pos:col_pos + 3])
            tmp = structure.loc[1:, cols].copy().dropna()
            tmp.columns = ["attributes", "type", "optional"]
            self.data_structure[c] = tmp

        structure = xls.parse("model")
        columns = [c for c in structure.columns if "Unnamed:" not in c]
        self.model_structure = {}
        for c in columns:
            col_pos = structure.columns.get_loc(c)
            cols = list(structure.columns[col_pos:col_pos + 3])
            tmp = structure.loc[1:, cols].copy()
            tmp = tmp.loc[~tmp.isna().all(axis=1), :]
            tmp.columns = ["attributes", "type", "default"]
            self.model_structure[c] = tmp

    def validate_modeldata(self):
        """Validate the processed input data to be conform with predifen model data structure.

        Similaritly to the :meth:`~validate_inputdata` method, this methods checks the consistency
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
                setattr(self, data, pd.DataFrame(columns=self.model_structure[data].attributes[1:]))
                self.logger.warning("%s not in Input Data, initialized as empty", data)
            else:
                tmp = getattr(self, data)
                for attr in [col for col in self.model_structure[data].attributes[1:] if col not in tmp.columns]:
                    default_value = self.model_structure[data].set_index("attributes").loc[attr, "default"]
                    tmp.loc[:, attr] = default_value
                    self.logger.warning("Attribute %s not in %s, initialized as %s", attr, data, str(default_value))
                setattr(self, data, tmp)

    def process_results(self, result_folder, grid=None):
        """Initialize :class:`~pomato.data.ResultProcessing` with `results_folder` and the own instance."""
        self.results = ResultProcessing(self, result_folder, grid)

    def return_results(self, symb):
        """Interface method to allow access to results from :class:`~pomato.data.ResultProcessing`."""
        if self.results and symb in self.result.__dict__.keys():
            return_value = getattr(self.results, symb)
        else:
            if not self.results:
                self.logger.error("Results not Initialized")
            else:
                self.logger.error("Symbol not in in results class")
            return_value = None
        return return_value

    def _clear_all_data(self):
        attr = list(self.__dict__.keys())
        attr.remove('logger')
        for att in attr:
            delattr(self, att)

    def visulize_inputdata(self, folder, show_plot=True):
        """Create default Plots for Input Data.

        This methods is currently not maintained, but was thought to provide standard figures
        to visualize the (processed) input data.
        """
        if not Path.is_dir(folder):
            self.logger.warning("Folder %s does not exist!", folder)
            self.logger.warning("Creating %s", folder)
            Path.mkdir(folder)
        if show_plot:
            plt.ion()
        else:
            plt.ioff()
        # Demand by Zone
        demand_zonal = pd.DataFrame(index=self.demand_el.index)
        for zone in self.zones.index:
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
            demand_zonal[zone] = self.demand_el[nodes_in_zone].sum(axis=1)
        fig_demand, ax_demand = plt.subplots()
        demand_zonal.plot.area(ax=ax_demand, xticks=np.arange(0, len(demand_zonal.index), step=10))
        ax_demand.legend(loc='upper right')
        ax_demand.margins(x=0)
        fig_demand.savefig(str(folder.joinpath("zonal_demand.png")))

        # Plot Installed Capacity by....
        plants_zone = pd.merge(self.plants, self.nodes.zone,
                               how="left", left_on="node", right_on="index")

        for elm in ["fuel", "tech"]:
            inst_capacity = plants_zone[["g_max", "zone", elm]].groupby([elm, "zone"],
                                                                        as_index=False).sum()
            fig_gen, ax_gen = plt.subplots()
            inst_capacity.pivot(index="zone",
                                columns=elm,
                                values="g_max").plot.bar(stacked=True, ax=ax_gen)

            ax_gen.legend(loc='upper right')
            ax_gen.margins(x=0)
            fig_gen.savefig(str(folder.joinpath(f"installed_capacity_by_{elm}.png")))
