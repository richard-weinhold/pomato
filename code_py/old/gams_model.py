"""
GAMS Model - Market Model for GRID Model
"""
import logging
import sys
import datetime as dt
import pandas as pd
import gams

class GamsModel(object):
    """Class of GAMS Model"""
    def __init__(self, wdir, DATA, GMS_SETUP, grid_rep, model_horizon):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.GamsModel')

        self.logger.info("Initializing MarketModel..")
        self.wdir = wdir
        self.workspace = gams.GamsWorkspace(working_directory=self.wdir.joinpath("gams"))
        self.gams_db = self.workspace.add_database()
        self.gams_result_db = self.workspace.add_database()

        self.model_type = GMS_SETUP["opt"]

        self.model_horizon = ['t'+ "{0:0>4}".format(x) for x in model_horizon]
        self.nodes = DATA.nodes
        self.zones = DATA.zones
        self.lines = DATA.lines
        self.plants = DATA.plants
        self.heatareas = DATA.heatareas
        self.demand_el = DATA.demand_el
        self.demand_h = DATA.demand_h
        self.availability = DATA.availability
        self.dclines = DATA.dclines

        self.init_gamsdb(grid_rep)
        self.check_domain_violations()

        self.gams_model = self.init_gams_model(GMS_SETUP)
        self.logger.info("MarketModel initialized!")

    def manually_add_result_db(self, gdx_path, gdx_name):
        """to avoid re-run - initialize result db without running the model"""
        self.gams_result_db = self.workspace.add_database_from_gdx(gdx_path + gdx_name)

    def check_domain_violations(self):
        """if there are domain violation this check will find&report them"""
        list_domain_violations = self.gams_db.get_database_dvs()
        if list_domain_violations != []:
            self.logger.error("GAMS db contains domain violations in the following parameters:")
            for i in list_domain_violations:
                self.logger.error(i.symbol.get_name())

    def init_gams_model(self, gms_setup):
        """add gams model to gams workspace from file gams_dispatch_python.gms"""

        ## add job from string because that way the different solve statements
        ## are included based on option and (possibly) different reporting files
        with open(self.wdir.joinpath("gams").joinpath("gams_dispatch_python.gms")) as gms_file:
            model_text = gms_file.read()

        # Force Infeasible Variables to 0, depending on model_type
        infeas = ''
        if not gms_setup["infeas_heat"]:
            infeas += "\nINFEAS_H_POS.fx(ha,t) = 0;\nINFEAS_H_NEG.fx(ha,t) = 0;"
        if not gms_setup["infeas_el"]:
            infeas += "\nINFEAS_EL_POS.fx(n,t) = 0;\nINFEAS_EL_NEG.fx(n,t) = 0;"
        if not (gms_setup["infeas_lines"]): #and gms_setup["opt"] in ["nodal", "cbco_nodal", "cbco_zonal"]):
            infeas += "\nINFEAS_LINES.fx(cb,t) = 0;\n"


        solve_statement = "\nsolve Model_" + self.model_type + " min COST use lp;\n"
        statistic = "\nms=Model_" + self.model_type + ".modelstat; ss=Model_" + \
                            self.model_type + ".solvestat;\n"

        gams_model = self.workspace.add_job_from_string(model_text + infeas + \
                                                        solve_statement + statistic)

        return gams_model

    def init_gamsdb(self, grid_rep):
        """Initializes gams database based on grid representation provided by grid model"""
        self.logger.info("Initializing GamsDatabase..")
        t = self.gams_db.add_set("t", 1, "Hour")
        for elm in self.demand_el.index[self.demand_el.index.isin(self.model_horizon)]:
            t.add_record(str(elm))
            # map_tdm.add_record([d.index, demand_el.day[d], demand_el.month[d]])

        p = self.gams_db.add_set("p", 1, "plants")
        for elm in self.plants.index:
            p.add_record(elm)

        co = self.gams_db.add_set("co", 1, "Conventional Plants of p")
        for elm in self.plants.index[(self.plants.g_max > 0)&(self.plants.tech != "dem")]:
            co.add_record(elm)

        he = self.gams_db.add_set("he", 1, "Heat Plants of p")
        for elm in self.plants.index[(self.plants.h_max > 0)]:
            he.add_record(elm)

        chp = self.gams_db.add_set("chp", 1, "CHP Plants of p")
        for elm in self.plants.index[(self.plants.h_max > 0) & (self.plants.g_max > 0)]:
            chp.add_record(elm)

        ph_tech = ["heatpump", "elheat"]
        ph = self.gams_db.add_set("ph", 1, "Power to Heat of p")
        for elm in self.plants.index[self.plants.tech.isin(ph_tech)]:
            ph.add_record(elm)

        hs = self.gams_db.add_set("hs", 1, "Heat Storage")
        for elm in self.plants.index[(self.plants.tech == 'storage') & (self.plants.g_max == 0)]:
            hs.add_record(elm)

        es = self.gams_db.add_set("es", 1, "Electricity Storage")
        for elm in self.plants.index[(self.plants.tech == 'storage') & (self.plants.h_max == 0)]:
            es.add_record(elm)

        ts_tech = ['pv', 'wind_off', 'wind_on', "pvheat", "iheat"]
        ts = self.gams_db.add_set("ts", 1, "Generation plants using Time-Series for Capacity of p ")
        for elm in self.plants.index[self.plants.tech.isin(ts_tech)]:
            ts.add_record(elm)

        d = self.gams_db.add_set("d", 1, "Demand Units of p")
        for elm in self.plants.index[self.plants.tech == 'dem']:
            d.add_record(elm)

        n = self.gams_db.add_set("n", 1, "nodes")
        for node in self.nodes.index:
            n.add_record(node)

        z = self.gams_db.add_set("z", 1, "zones Electric")
        for zone in self.zones.index:
            z.add_record(zone)

        ha = self.gams_db.add_set("ha", 1, "Heat Area")
        for elm in self.heatareas.index:
            ha.add_record(elm)

        map_pn = self.gams_db.add_set("map_pn", 2, "Plant-Node Mapping")
        for plant in self.plants.index:
            map_pn.add_record([plant, self.plants.node[plant]])

        map_nz = self.gams_db.add_set("map_nz", 2, "Node-Zone El Mapping")
        for node in self.nodes.index:
            map_nz.add_record([node, self.nodes.zone[node]])

        map_pha = self.gams_db.add_set("map_pha", 2, "Plant-Heat Area Mapping")
        for plant in self.plants.index[self.plants.heatarea.notnull()]:
            map_pha.add_record([plant, self.plants.heatarea[plant]])

        ## Init Parameter
        mc = self.gams_db.add_parameter_dc("mc", [p], "Marginal Costs of Plant p")
        g_max = self.gams_db.add_parameter_dc("g_max", [p], "Maximum Generation of Plant p")
        eta = self.gams_db.add_parameter_dc("eta", [p], "Efficiency")
        for plant in self.plants.index:
            mc.add_record(plant).value = float(self.plants.mc[plant])
            g_max.add_record(plant).value = float(self.plants.g_max[plant])
            eta.add_record(plant).value = float(self.plants.eta[plant])

        h_max = self.gams_db.add_parameter_dc("h_max", [p], "Maximum Heat-Generation of Plant h(p)")
        for plant in self.plants.index[self.plants.h_max > 0]:
            h_max.add_record(plant).value = float(self.plants.h_max[plant])

        ## Storage Capacity (based on maximum generation))
        es_cap = self.gams_db.add_parameter_dc("es_cap", [es], "Electricity Storage Capacity")
        for elm in self.gams_db['es']:
            es_cap.add_record(elm.keys[0]).value = float(self.plants.g_max[elm.keys[0]]*4)

        hs_cap = self.gams_db.add_parameter_dc("hs_cap", [hs], "Heat Storage Capacity")
        for elm in self.gams_db['hs']:
            hs_cap.add_record(elm.keys[0]).value = float(self.plants.h_max[elm.keys[0]]*4)

        ### Assining All timedependant parameters
        d_el = self.gams_db.add_parameter_dc("d_el", [n, t],
                                             "Electricity demand at node n")
        d_h = self.gams_db.add_parameter_dc("d_h", [ha, t],
                                            "Heat demand at node n")
        ava = self.gams_db.add_parameter_dc("ava", [p, t],
                                            "availability of Time-Series Dependant Generation")
        for time in self.gams_db['t']:
            for node in self.nodes.index:
                d_el.add_record([node, time.keys[0]]).value = \
                                      float(self.demand_el[node][time.keys[0]])
            for elm in self.heatareas.index:
                d_h.add_record([elm, time.keys[0]]).value = \
                                        float(self.demand_h[elm][time.keys[0]])
            for elm in self.gams_db['ts']:
                ava.add_record([elm.keys[0], time.keys[0]]).value =\
                            float(self.availability[elm.keys[0]][time.keys[0]])
            for elm in self.gams_db['d']:
                ava.add_record([elm.keys[0], time.keys[0]]).value =\
                            float(self.availability[elm.keys[0]][time.keys[0]])

        ## definition of grid stuff, however empty
        cb = self.gams_db.add_set("cb", 1, "Critical Branch")
        dc = self.gams_db.add_set("dc", 1, "DC lines")
        l = self.gams_db.add_set("l", 1, "lines")
        slack = self.gams_db.add_set("slack", 1, "Slack Node")
        map_ns = self.gams_db.add_set("map_ns", 2, "Node to Slack mapping")

        ptdf = self.gams_db.add_parameter_dc("ptdf", [n, l],
                                             "PTDF Matrix - Node-Line Sensitivilty")
        ntc = self.gams_db.add_parameter_dc("ntc", [z, z], "NTC between two zones")

        inc_dc = self.gams_db.add_parameter_dc("inc_dc", [dc, n],
                                               "Incedence Matrix for DC Connections")
        dc_max = self.gams_db.add_parameter_dc("dc_max", [dc],
                                               "Maximum transmission capacity of DC Line")
        l_max = self.gams_db.add_parameter_dc("l_max", [l], "Line Themal Capacity")
        ram = self.gams_db.add_parameter_dc("ram", [cb],
                                            "Remaining Available Margin on CB c")

        # define everything - left empty if not needed
        if self.model_type == "ntc":
            for ntc_index in grid_rep["ntc"].index:
                zone_i = grid_rep["ntc"].zone_i[ntc_index]
                zone_j = grid_rep["ntc"].zone_j[ntc_index]
                ntc.add_record([zone_i, zone_j]).value = \
                                        float(grid_rep["ntc"].ntc[ntc_index])

        if self.model_type == "nodal":
            for line in self.lines.index:
                l.add_record(line)
            for node_index, node in enumerate(self.nodes.index):
                for line_index, line in enumerate(self.lines.index):
                    ptdf.add_record([node, line]).value = grid_rep["ptdf"][line_index, node_index]
            for line in self.lines.index:
                l_max.add_record(line).value = float(self.lines.maxflow[line])

        if "cbco" in self.model_type.split("_"):
            cbco_data = {}
            for i in grid_rep["cbco"]:
                cbco_data['cb'+ "{0:0>3}".format(i)] = \
                            {'cbco':  list(grid_rep["cbco"][i]['ptdf']), \
                             'ram': grid_rep["cbco"][i]['ram']}
            for i in cbco_data:
                cb.add_record(i)

        if self.model_type == 'cbco_nodal':
            cbco = self.gams_db.add_parameter_dc("cbco", [cb, n], \
                                                 "Critical Branch Critical Outage - \
                                                 for Zone to Line Sensitivities ")
            for cb in self.gams_db['cb']:
                ram.add_record(cb.keys[0]).value = float(cbco_data[cb.keys[0]]['ram'])
                for j, node in enumerate(self.nodes.index):
                    cbco.add_record([cb.keys[0], node]).value = \
                                    float(cbco_data[cb.keys[0]]['cbco'][j])
        elif self.model_type == 'cbco_zonal':
            cbco = self.gams_db.add_parameter_dc("cbco", [cb, z], \
                                                 "Critical Branch Critical Outage - \
                                                 for Zone to Line Sensitivities ")
            for cb in self.gams_db['cb']:
                ram.add_record(cb.keys[0]).value = \
                                float(cbco_data[cb.keys[0]]['ram'])
                for j, zone in enumerate(self.zones.index):
                    cbco.add_record([cb.keys[0], zone]).value = \
                                    float(cbco_data[cb.keys[0]]['cbco'][j])
        else:
            cbco = self.gams_db.add_parameter_dc("cbco", [cb, n], \
                                     "Critical Branch Critical Outage - \
                                     for Zone to Line Sensitivities ")

        for node in self.nodes.index[self.nodes.slack]:
            slack.add_record(node)
            for elm in grid_rep["slack_zones"][node]:
                map_ns.add_record([str(elm), str(node)])

        for dcline in self.dclines.index:
            dc.add_record(dcline)
            dc_max.add_record(dcline).value = float(self.dclines.capacity[dcline])
            inc_dc.add_record([str(dcline), str(self.dclines.node_i[dcline])]).value = 1
            inc_dc.add_record([str(dcline), str(self.dclines.node_j[dcline])]).value = -1

        self.logger.info("GamsDatabase initialized!")

    def run(self):
        """ Run GAMS Model - includes gams option settings"""
        ## Define options
        opt = self.workspace.add_options()
        opt.defines["gdxincname"] = self.gams_db.name
        opt.defines["model_option"] = self.model_type
        opt.profile = 1
        opt.limcol = 0
        opt.limrow = 0
        opt.solprint = 0
        opt.threads = 0
        opt.optfile = 1

        with open(self.wdir.joinpath("cplex.opt"), "w") as cplex_optfile:
            cplex_optfile.write("names=1")

        self.logger.info("Run Model! GridRepresentation: " + self.model_type + "\n")
        self.gams_model.run(opt, databases=self.gams_db, output=sys.stdout)
        self.gams_result_db = self.gams_model.out_db
        # Check Model Stats
        self.logger.info('Solvestat= ' + \
                         str(self.gams_model.out_db['ss'].find_record().value) \
                         +'; Modelstat= ' + \
                         str(self.gams_model.out_db['ms'].find_record().value))

        if self.gams_model.out_db['ss'].find_record().value == 1 \
        and self.gams_model.out_db['ms'].find_record().value == 1:
            objective_value = self.gams_model.out_db['COST'].first_record().level
            self.logger.info(f'Optimal -> Objective Value: {objective_value}\n')
        else:
            self.logger.warning('Not Optimal -> check gams lst file for more information\n')

        output_name = self.model_type + "_" + dt.datetime.now().strftime("%d%m_%H%M") + ".gdx"
        # if gdxfiles folder doesnt exist -> create
        if not self.wdir.joinpath("gdxfiles").is_dir():
            self.wdir.joinpath("gdxfiles").mkdir()
            self.logger.info("Created /gdxfiles for gdx export")

        self.gams_model.out_db.export(str(self.wdir.joinpath("gdxfiles").joinpath(output_name)))
        self.logger.info("gdx file: " + output_name + " saved to /gdxfiles folder")
        self.check_for_infeas()
        self.logger.info("Market Model complete!\n\n")
        ## Set OutDb to result db

    def return_inj_t(self, timestep):
        """ returns only nodal net injetions as list for t as str"""
        inj = []
        for node in self.nodes.index:
            inj.append(self.gams_result_db["INJ"].find_record(keys=[node, timestep]).level)
        return inj

    def check_for_infeas(self):
        """
        checks for infeasiblities in electricity/heat energy balances
        returns nothing
        """
        self.logger.info("Check for infeasiblities in electricity energy balance...")
        infeas_pos = self.gams_symbol_to_df("INFEAS_EL_POS")
        infeas_pos = infeas_pos[infeas_pos.INFEAS_EL_POS != 0]
        infeas_neg = self.gams_symbol_to_df("INFEAS_EL_NEG")
        infeas_neg = infeas_neg[infeas_neg.INFEAS_EL_NEG != 0]

        if not (infeas_pos.empty and infeas_neg.empty):
            nr_n = len(infeas_neg.groupby(["n"]).count())
            nr_t = len(infeas_neg.groupby(["t"]).count())
            self.logger.warning("Negative infeasibilities in " + str(nr_t) +
                                " timesteps and at " + str(nr_n) + " different nodes")
            nr_n = len(infeas_pos.groupby(["n"]).count())
            nr_t = len(infeas_pos.groupby(["t"]).count())
            self.logger.warning("Positive infeasibilities in " + str(nr_t) +
                                " timesteps and at " + str(nr_n) + " different nodes")

        self.logger.info("Check for infeasiblities in heat energy balance...")
        infeas_pos = self.gams_symbol_to_df("INFEAS_H_POS")
        infeas_pos = infeas_pos[infeas_pos.INFEAS_H_POS != 0]
        infeas_neg = self.gams_symbol_to_df("INFEAS_H_NEG")
        infeas_neg = infeas_neg[infeas_neg.INFEAS_H_NEG != 0]

        if not (infeas_pos.empty and infeas_neg.empty):
            nr_n = len(infeas_neg.groupby(["ha"]).count())
            nr_t = len(infeas_neg.groupby(["t"]).count())
            self.logger.warning("Negative infeasibilities in " + str(nr_t) +
                                " timesteps and at " + str(nr_n) + " different nodes")
            nr_n = len(infeas_pos.groupby(["ha"]).count())
            nr_t = len(infeas_pos.groupby(["t"]).count())
            self.logger.warning("Positive infeasibilities in " + str(nr_t) +
                                " timesteps and at " + str(nr_n) + " different nodes")
        self.logger.info("Check for infeasiblities on Lines...")
        infeas_lines = self.gams_symbol_to_df("INFEAS_LINES")
        infeas_lines = infeas_lines[infeas_lines.INFEAS_LINES != 0]
        if not infeas_lines.empty:
            nr_cb = len(infeas_lines.groupby(["cb"]).count())
            nr_t = len(infeas_lines.groupby(["t"]).count())
            self.logger.warning("Infeasibilities in " + str(nr_t) +
                                " timesteps and at " + str(nr_cb) + " different cbcos")

    def gams_symbol_to_df(self, symb):
        """Returns DataFrame for any GamsVariable, GamsParameter or GamsSet"""
        columns = self.gams_result_db[symb].domains_as_strings + [symb]
        if isinstance(self.gams_result_db[symb], gams.GamsParameter):
            list_gmssymbol = []
            for i in self.gams_result_db[symb]:
                list_gmssymbol.append(i.keys + [i.value])
            dataframe = pd.DataFrame(columns=columns, data=list_gmssymbol)
        elif isinstance(self.gams_result_db[symb], gams.GamsVariable):
            list_gmssymbol = []
            for i in self.gams_result_db[symb]:
                list_gmssymbol.append(i.keys + [i.level])
            dataframe = pd.DataFrame(columns=columns, data=list_gmssymbol)
        elif isinstance(self.gams_result_db[symb], gams.GamsSet):
            list_gmssymbol = []
            for i in self.gams_result_db[symb]:
                list_gmssymbol.append(i.keys + [True])
            dataframe = pd.DataFrame(columns=columns, data=list_gmssymbol)
        elif isinstance(self.gams_result_db[symb], gams.GamsEquation):
            list_gmssymbol = []
            for i in self.gams_result_db[symb]:
                list_gmssymbol.append(i.keys + [i.level, i.marginal])
            columns = self.gams_result_db[symb].domains_as_strings + \
                      [symb + "_level", symb + "_marginal"]
            dataframe = pd.DataFrame(columns=columns, data=list_gmssymbol)
        else:
            self.logger.error("Please Specify valid gamsVariable or gamsParameter")
        return dataframe

    def nodal_prices(self):
        """returns nodal electricity price"""
        eb_nodal = self.gams_symbol_to_df("EB_Nodal")
        eb_nodal.columns = ["n", "t", "l", "m_n"]
        eb_nodal = pd.merge(eb_nodal, self.nodes.zone.to_frame(),
                            how="left", left_on="n", right_index=True)
        eb_nodal = eb_nodal.drop("l", axis=1)
        eb_nodal.m_n[abs(eb_nodal.m_n) < 1E-3] = 0

        eb_zonal = self.gams_symbol_to_df("EB_Zonal")
        eb_zonal.columns = ["z", "t", "l", "m_z"]
        eb_zonal = eb_zonal.drop("l", axis=1)
        eb_zonal.m_z[abs(eb_zonal.m_z) < 1E-3] = 0

        price = pd.merge(eb_nodal, eb_zonal, how="left",
                         left_on=["t", "zone"], right_on=["t", "z"])
        price["marginal"] = -(price.m_z + price.m_n)
        return price[["t", "n", "z", "marginal"]]

    def total_demand(self):
        """Returns DataFrame with all relevant Demand as timeseries"""
        demand_d = self.gams_symbol_to_df("D_d")
        demand_ph = self.gams_symbol_to_df("D_ph")
        demand_es = self.gams_symbol_to_df("D_es")
        map_dn = self.gams_symbol_to_df("map_pn")
        demand_d = pd.merge(demand_d, map_dn[["p", "n"]],
                            how="left", left_on="d", right_on="p")
        if not demand_d.empty:
            demand_d = demand_d.groupby(["n", "t"], as_index=False).sum()
        demand_ph = pd.merge(demand_ph, map_dn[["p", "n"]],
                             how="left", left_on="ph", right_on="p")
        if not demand_ph.empty:
            demand_ph = demand_ph.groupby(["n", "t"], as_index=False).sum()
        demand_es = pd.merge(demand_es, map_dn[["p", "n"]],
                             how="left", left_on="es", right_on="p")
        if not demand_es.empty:
            demand_es = demand_es.groupby(["n", "t"], as_index=False).sum()

        demand = self.gams_symbol_to_df("d_el")
        demand = pd.merge(demand, demand_d[["D_d", "n", "t"]],
                          how="outer", on=["n", "t"])
        demand = pd.merge(demand, demand_ph[["D_ph", "n", "t"]],
                          how="outer", on=["n", "t"])
        demand = pd.merge(demand, demand_es[["D_es", "n", "t"]],
                          how="outer", on=["n", "t"])
        demand.fillna(value=0, inplace=True)

        demand["d_total"] = demand.d_el + demand.D_d + demand.D_ph + demand.D_es
        return demand

    def plot_generation_area(self, option="fuel"):
        """ Plot Generatin as Area Plot by technology or fuel"""
        df_g = self.gams_symbol_to_df("G")
        tmp = pd.merge(df_g, self.plants[["tech", "fuel_mix"]],
                       how="left", left_on="p", right_index=True)
        if option == "tech":
            tec = tmp.groupby(["t", "tech"], as_index=False).sum()
            tec = tec.pivot(index="t", columns='tech')['G']
            tec.plot.area()
        else:
            fuel = tmp.groupby(["t", "fuel_mix"], as_index=False).sum()
            fuel = fuel.pivot(index="t", columns='fuel_mix')['G']
            fuel.plot.area()

    def return_results(self, symb):
        """ Interface Method to allow for analog access to data as in the julia interface"""
        return self.gams_symbol_to_df(symb)