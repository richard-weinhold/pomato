
"""
GAMS CBCO Reduction Algorithm
"""
#import gams
import numpy as np
import datetime
import json

class LPReduction(object):
    """Class of GAMS Model"""
    def __init__(self, wdir, A_matrix, b_vector):
        print("Initializing CBCO Reduction Method..")
        self.wdir = wdir
        self.workspace = gams.GamsWorkspace(working_directory=self.wdir.joinpath("gams"))
        self.gams_db = self.workspace.add_database()
        if isinstance(A_matrix, list):
            self.A_matrix = np.array(A_matrix)
            self.b_vector = np.array(b_vector).reshape(len(b_vector), 1)
        else:
           self.A_matrix = A_matrix
           self.b_vector = b_vector

        self.gams_model = self.init_gams_model()

        self.row_index = self.gams_db.add_set("row_index", 1, "row_index")
        self.col_index = self.gams_db.add_set("col_index", 1, "col_index")

        for col in range(0, np.shape(self.A_matrix)[1]):
            self.col_index.add_record(str(col))

        self.A = self.gams_db.add_parameter_dc("A", [self.row_index, self.col_index], "A Matrix (LHS)")
        self.b = self.gams_db.add_parameter_dc("b", [self.row_index], "b vector (RHS)")
        self.s = self.gams_db.add_parameter_dc("s", [self.col_index], "additional row for A   ")
        self.t = self.gams_db.add_parameter("t", 0)

        # add first element
        self.row_index_counter = len(self.b_vector) - 1
        self.cbco_rows = [x for x in range(0, len(self.b_vector))]
        self.update_Ab()

        print("Reduction Algorithm Init!")


    def save_cbco_to_json(self):
        """saves the final list of indexes of linear constraints to json"""
        filename = f'reduced_cbco_{datetime.datetime.now().strftime("%d%m_%H%M")}.json'
        with open(self.wdir.joinpath(filename), 'w') as file:
                json.dump(self.cbco_rows, file)

    def feasible_point(self):
        self.row_index_counter = len(self.b_vector)
        self.update_Ab()
        self.s.clear()
        self.t.clear()
        for col in range(0, np.shape(self.A_matrix)[1]):
            self.s.add_record(str(col)).value = float(1)
        self.t.add_record().value = float(1000)
        self.run()
        result = {}
        for i in self.gams_model.out_db["x"]:
            result[i.keys[0]] = i.level
        return result

    def algorithm(self):
        """actual algorithm"""
        for i in range(0, np.shape(self.A_matrix)[0]):
            self.update_st()
            self.run()
            if self.gams_model.out_db['z'].first_record().level < float(self.b_vector[self.row_index_counter]):
                self.cbco_rows.remove(self.row_index_counter)
            self.row_index_counter -= 1
            self.update_Ab()
        print("cbco counter: ", len(self.cbco_rows))
        print("Saved cbco indecies to json")
        self.save_cbco_to_json()

    def init_gams_model(self):
        """add gams model to gams workspace from file cbco_reduction.gms"""
        with open(self.wdir.joinpath("gams").joinpath("cbco_reduction.gms")) as gms_file:
            gams_model = self.workspace.add_job_from_string(gms_file.read())

        with open(self.wdir.joinpath("cplex.opt"), "w") as cplex_optfile:
            cplex_optfile.write("names=1")

        return gams_model

    def update_st(self):
        """Next row in s*x < t"""
        self.s.clear()
        self.t.clear()
        for col in range(0, np.shape(self.A_matrix)[1]):
            self.s.add_record(str(col)).value = float(self.A_matrix[self.row_index_counter, col])
        self.t.add_record().value = float(self.b_vector[self.row_index_counter])

    def update_Ab(self):
        self.row_index.clear()
        self.A.clear()
        self.b.clear()
        for row in self.cbco_rows:
            if row != self.row_index_counter:
                self.row_index.add_record(str(row))
                self.b.add_record(str(row)).value = float(self.b_vector[row])
        for row in self.cbco_rows:
            if row != self.row_index_counter:
                for col in range(0, np.shape(self.A_matrix)[1]):
                    self.A.add_record([str(row), str(col)]).value = \
                                      float(self.A_matrix[row,col])

    @property
    def options(self):
        opt = self.workspace.add_options()
        opt.defines["gdxincname"] = self.gams_db.name
        opt.profile = 1
        opt.limcol = 0
        opt.limrow = 0
        opt.solprint = 0
        opt.threads = 0
        opt.optfile = 1
        return opt

    def run(self):
        self.gams_model.run(self.options, databases=self.gams_db) #, output=sys.stdout)
        # Check Model Stats
        if self.gams_model.out_db['ss'].find_record().value != 1 or \
           self.gams_model.out_db['ms'].find_record().value != 1:
                print("Not optimal with row_index:", self.row_index_counter)
        else:
            print(self.row_index_counter)
