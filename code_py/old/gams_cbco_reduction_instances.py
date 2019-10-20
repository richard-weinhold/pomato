# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:51:33 2018

@author: Richard Weinhold
"""

"""
GAMS Model - Market Model for GRID Model
"""
import gams
import numpy as np
class LPReduction(object):
    """Class of GAMS Model"""
    def __init__(self, wdir, A_matrix, b_vector):

#        A_matrix = A
#        b_vector = b
#
#        self = test

        print("Initializing CBCO Reduction Method..")
        self.wdir = wdir
        self.workspace = gams.GamsWorkspace(working_directory=self.wdir)
        self.gams_db = self.workspace.add_database()

        if isinstance(A_matrix, list):
            self.A_matrix = np.array(A_matrix)
            self.b_vector = np.array(b_vector).reshape(len(b_vector), 1)
        else:
           self.A_matrix = A_matrix
           self.b_vector = b_vector

        self.row_index = self.gams_db.add_set("row_index", 1, "row_index")
        self.col_index = self.gams_db.add_set("col_index", 1, "col_index")
#        self.cbco_index = self.gams_db.add_set("cbco_index", 1, "cbco_index")

        for col in range(0, np.shape(self.A_matrix)[1]):
            self.col_index.add_record(str(col))

        self.A = self.gams_db.add_parameter_dc("A", [self.row_index, self.col_index], "A Matrix (LHS)")
        self.b = self.gams_db.add_parameter_dc("b", [self.row_index], "b vector (RHS)")

#        self.s = self.gams_db.add_parameter_dc("s", [self.col_index], "additional row for A   ")
#        self.t = self.gams_db.add_parameter("t", 0)

        # add first element
        self.row_index_counter = len(self.b_vector) - 1
        self.cbco_rows = [x for x in range(0, len(self.b_vector))]

        ## Full A Matrix, full b vetor, full row_index
        for row in self.cbco_rows:
            self.row_index.add_record(str(row))
            self.b.add_record(str(row)).value = float(self.b_vector[row])
        for row in self.cbco_rows:
            for col in range(0, np.shape(self.A_matrix)[1]):
                self.A.add_record([str(row), str(col)]).value = \
                                  float(self.A_matrix[row,col])


        self.s = self.gams_db.add_parameter("s", 1, "additional row for A")
        self.t = self.gams_db.add_parameter("t", 0, "t value")
        self.cbco_index = self.gams_db.add_parameter("cbco_index", 1, "cbco_index")

        self.s, self.t, self.cbco_index = self.update_s_t_cbco_index(self.s, self.t, self.cbco_index)

        self.gams_model, self.model_instance = self.init_gams_model()

        self.s = self.model_instance.sync_db.add_parameter("s", 1, "additional row for A")
        self.t = self.model_instance.sync_db.add_parameter("t", 0, "t value")
        self.cbco_index = self.model_instance.sync_db.add_parameter("cbco_index", 1, "cbco_index")

        print("Reduction Algorithm Init!")

    def algorithm(self):
        """actual algorithm"""
#        self.update_s_t_cbco_index()
        self.model_instance.instantiate("cbco use lp max z;",
                                        [gams.GamsModifier(self.s), gams.GamsModifier(self.t), gams.GamsModifier(self.cbco_index)],
                                        self.options)

        for i in range(0, np.shape(self.A_matrix)[0]):
            self.s, self.t, self.cbco_index = self.update_s_t_cbco_index(self.s, self.t, self.cbco_index)
            self.run()
            print("  Modelstatus: " + str(self.model_instance.model_status))
            print("  Solvestatus: " + str(self.model_instance.solver_status))
            print("  Obj: " + str(self.model_instance.sync_db.get_variable("z").find_record().level))

            if self.gams_model.out_db['z'].first_record().level < float(self.b_vector[self.row_index_counter]):
                del self.cbco_rows[self.row_index_counter]
            self.row_index_counter -= 1

        print("cbco counter: ", len(self.cbco_rows))

    def init_gams_model(self):
        """add gams model to gams workspace from file gams_lp_cbco.gms"""
        model_text = open(self.wdir + "\\cbco_reduction_instances.gms")

        cp = self.workspace.add_checkpoint()
        gams_model = self.workspace.add_job_from_string(model_text.read())
        model_text.close()

        gams_model.run(self.options, databases=self.gams_db, checkpoint=cp)
        model_instance = cp.add_modelinstance()

        cplex_optfile = open(self.wdir + "\\cplex.opt", "w", buffering=-1, encoding="utf-8")
        cplex_optfile.write("names=1")
        cplex_optfile.close()

        return gams_model, model_instance

    def update_s_t_cbco_index(self, s, t, cbco_index):
        """Next row in s*x < t"""
        s.clear()
        t.clear()
        cbco_index.clear()

        for col in range(0, np.shape(self.A_matrix)[1]):
            s.add_record(str(col)).value = float(self.A_matrix[self.row_index_counter, col])

        t.add_record().value = float(self.b_vector[self.row_index_counter])

        for cbco in self.cbco_rows:
            cbco_index.add_record(str(cbco)).value = 1

        return s, t, cbco

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
        self.model_instance.run(self.options) #, output=sys.stdout)
        # Check Model Stats
        if self.model_instance.solver_status != 1 or \
           self.model_instance.model_status != 1:
                print("Not optimal with row_index:", self.row_index_counter)
        else:
            print(self.row_index_counter)
