# import sys
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt

import pomato.tools as tools
from pomato.data import ResultProcessing
# pylint: disable-msg=E1101


class ENSResultProcessing(ResultProcessing):

    def __init__(self, data, grid, result_folder):
        super().__init__(data, grid, result_folder)

        xls = pd.ExcelFile(self.data.wdir.joinpath(self.data.data_source))

        self.fuelmix = xls.parse("fuelmix", index_col=0)
        self.generation_costs = xls.parse("generation_costs", index_col=0)

        self.dump_all_results()
        
    def dump_all_results(self):
    	pass

    def calculate_emissions(self):
        """Emissions from the ramses emissions factors co2 in kg/GJ and so2 in g/GJ

        Returns
        -------
        co2: pd.Series
            CO2 emissions per plant for the modeled model horizon in t
        so2: pd.Series
            SO2 emissions per plant for the modeled model horizon in t
        """

        emissions = pd.merge(self.data.plants[["fuel", "tech", "eta"]], self.G,
                             left_index=True, right_on="p", how="left")
        emissions = pd.merge(emissions, self.data.fuel,
                             left_on="fuel", right_index=True, how="left")

        emissions["co2_emissions"] = emissions["co2_content"] / emissions["eta"] * emissions["G"] * 3.6 / 1000
        emissions["so2_emissions"] = emissions["so2_content"] / emissions["eta"] * emissions["G"] * 3.6 / 1E6
        co2 = emissions[["p", "co2_emissions"]].groupby("p").sum()
        so2 = emissions[["p", "so2_emissions"]].groupby("p").sum()
        return co2, so2

    def calculate_fuel_consumption(self):

        fueluse = pd.merge(self.data.plants[["fuel", "tech", "eta"]], self.G, left_index=True,
                   right_on="p", how="left")
        fueluse = pd.merge(fueluse, self.fuelmix, on="fuel", how="left")
        fueluse["fueluse"] = (fueluse["G"] * fueluse["value"])/fueluse["eta"]
        fueluse_statistics = fueluse[["p", "content", "fueluse"]].groupby(["p", "content"]).sum().reset_index()
        fueluse_statistics = fueluse_statistics[fueluse_statistics.fueluse > 0]

        return fueluse_statistics


    def calculate_income(self):
        """Electricity spot and subsidy income"""

        income_g = pd.merge(self.data.plants[["node", "fuel"]], self.G, left_index=True,
                            right_on="p", how="right")

        income_dph = pd.merge(self.data.plants[["node", "fuel"]], self.D_ph, left_index=True,
                              right_on="p", how="right")

        income = pd.merge(income_g, income_dph, on=['node', 'fuel', 't', 'p'], how="outer").fillna(0)

        price = self.price()[["n", "t", "marginal"]]
        price.columns = ["node", "t", "price"]

        income = pd.merge(income, price, on=["node", "t"], how="left")
        income["spot_income"] = income["G"] * income["price"]

        income = pd.merge(income, self.generation_costs, right_index=True,
                          left_on="p", how="left")
        income["subsidy_income"] = (income["G"] - income["D_ph"]) * income["Subsidy"]

        return income[["subsidy_income", "spot_income", "p"]].groupby("p").sum()

    def calculate_costs(self):
        gen_cost = pd.merge(self.data.plants[["fuel", "tech"]], self.G, left_index=True,
                    right_on="p", how="right")

        gen_cost = pd.merge(gen_cost, self.generation_costs, left_on="p",
                    right_index=True, how="right")


        gen_cost["om_cost"] = gen_cost["G"]*gen_cost["mc_el"]
        gen_cost["fuel_cost"] = gen_cost["G"]*gen_cost["mc_el"]
        gen_cost["emission_cost"] = gen_cost["G"]*gen_cost["mc_el"]

        gen_cost["gencost"] = gen_cost["om_cost"] + gen_cost["fuel_cost"] + gen_cost["emission_cost"]

        return gen_cost[["p", "gencost"]].groupby("p").sum()

    def calculate_revenue(self):
        income = self.calculate_income()
        costs = self.calculate_costs()
        revenue = self.data.plants.copy()

        revenue = pd.merge(revenue, income, right_index=True, left_index=True, how="left")
        revenue = pd.merge(revenue, costs, right_index=True, left_index=True, how="left")
        revenue.loc[:, "spot_income"] = revenue.loc[:, "spot_income"].fillna(0)
        revenue.loc[:, "subsidy_income"] = revenue.loc[:, "subsidy_income"].fillna(0)
        revenue.loc[:, "gencost"] = revenue.loc[:, "gencost"].fillna(0)
        revenue["revenue"] = revenue["spot_income"] + revenue["subsidy_income"] - revenue["gencost"]
        return revenue[["revenue", "spot_income", "subsidy_income", "gencost"]]
