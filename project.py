# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:39:20 2025

@author: egeki
"""
import time

import pandas as pd
from pyomo.environ import *


model = ConcreteModel()
solver = SolverFactory('cplex')
solver.options['timelimit'] = 36000
solver.options['mipgap'] = 0.0

#file names needs to be implemented after datasets are given out
pallet_data = pd.read_excel("ProjectPart1-Scenario3.xlsx",sheet_name = "Pallets" ) 
vehicle_data = pd.read_excel("ProjectPart1-Scenario3.xlsx",sheet_name= "Vehicles")  
order_data = pd.read_excel("ProjectPart1-Scenario3.xlsx",sheet_name = "Orders")




#mapper for vehicles
vehicle_type_mapping={ 
    'Truck' : 1,
    'Lorry' : 2,
    'Van' : 3
}
Q = 15
#to be able to directly access vehicle capacities
vehicle_capacity ={
        (1,1): 22,
        (1,2) : 33,
        (2,1): 12,
        (2,2) : 18,
        (3,1): 6,
        (3,2) : 8,
}

#given on homework text we are on a range of atmost 7 so i initialize the range as 1-7 but could be read from csv aswell
model.DAYS = Set(initialize=range(1,8)) 
#need to be decided based on the type of datasets i will be given both days and products
model.PRODUCTS = Set(initialize=pallet_data['Product Type'].unique()) 
#only 3 types of vehicles are declared so we can hardcode
model.VEHICLE_TYPES = Set(initialize = [1,2,3]) 
#where 1 corresponds to 100x120 and 2 is 80x120 same as vehicle types we can hardcode since wont be more sizes
model.PALLET_SIZES = Set(initialize = [1,2]) 
#as named its for getting the capacity for given vehicle type with given size
model.capacity = Param(model.VEHICLE_TYPES, model.PALLET_SIZES,initialize=vehicle_capacity) 
#tracking cost of vehicles seperately
model.vehicle_cost = Param(model.VEHICLE_TYPES, mutable=True, initialize={})
model.vehicle_cost_rented =Param(model.VEHICLE_TYPES,mutable =True,initialize={})
#count of owned vehicles for our company so we can decide after what point they count as rental
model.vehicle_count =Param(model.VEHICLE_TYPES,mutable=True,initialize={})
#to hold operation count for each vehicle and day 


#to be able to update available product we need to be able to tell how much releases when
model.released_product = Param(model.PRODUCTS,model.PALLET_SIZES, model.DAYS, domain=NonNegativeIntegers,default = 0,mutable=True)  





# Initialize the set with standard Python integers
model.ORDERS = Set(initialize=order_data['Order ID'].unique()) 



# to be able to compare the demand with shipped amount to make sure we dont underdeliver
model.ordered_product = Param(model.ORDERS,model.PRODUCTS,  
                              initialize={(row['Order ID'], row['Product Type']): row['Demand Amount'] 
                                          for _, row in order_data.iterrows()}, 
                              default=0)

# to keep the due dates to be able to make the necessary comparation
model.due_date = Param(model.ORDERS,model.PRODUCTS,
                       initialize={(row['Order ID'], row['Product Type']): row['Due Date'] 
                                   for _, row in order_data.iterrows()},default =0)

# for the amount that is unnecessary but delivered we have to pay earliness penalty
model.earliness_penalty = Param(model.ORDERS,model.PRODUCTS, 
                                initialize={(row['Order ID'], row['Product Type']): row['Earliness Penalty'] 
                                            for _, row in order_data.iterrows()}, default=0)



model.owned_operations=Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers)
model.rented_operations=Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers)
model.operations = Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers,initialize=0) 
#the amount of vehicle assigned for the specific day, type and size 
model.vehicle_assigned = Var(model.DAYS,model.VEHICLE_TYPES,initialize =0,domain=NonNegativeIntegers)
#the amount of vehicle rented
model.vehicle_rented = Var(model.DAYS, model.VEHICLE_TYPES,domain=NonNegativeIntegers,initialize =0)
model.shipped_for_order=Var(model.ORDERS,model.PRODUCTS,model.DAYS,model.PALLET_SIZES,initialize=0,domain=NonNegativeIntegers)




#filling the necessary variables from the table vehicle_data
for index,row in vehicle_data.iterrows():
    vehicle_type = row["Vehicle Type"]
    model.vehicle_cost[vehicle_type] = row["Fixed Cost (c_k)"]
    model.vehicle_cost_rented[vehicle_type] = row["Variable Cost (c'_k)"]

# First, group the data by 'Vehicle Type' and sum the 'Num of vehicles'
vehicle_data_grouped = vehicle_data.groupby('Vehicle Type').size().reset_index(name='Num of vehicles')

# Now iterate over the grouped rows
for index, row in vehicle_data_grouped.iterrows():
    amount = row['Num of vehicles']
    vehicle_type = row['Vehicle Type']
    model.vehicle_count[vehicle_type] = amount
    


#grabbing the data from pallet_data table
for index, row in pallet_data.iterrows(): #here from the table we calculate released product count for each day for every size and product type
    product = row['Product Type']
    release_day = row['Release Day']
    amount = row['Amount']
    size_type = row['Pallet Size']
    model.released_product[product,size_type,release_day] += amount
    
  



def shipment_limit_rule(model, p, d):
    # Check if sum of shipped products for a product `p` until day `d` does not exceed released products
   return sum(model.shipped_for_order[o, p, day,1] for day in range(1,d+1)for o in model.ORDERS) <= sum(model.released_product[p, 1,day] for day in range(1,d+1))

model.ShipmentLimit = Constraint(model.PRODUCTS, model.DAYS, rule=shipment_limit_rule)


def shipment_limit_rule2(model, p, d):
    return sum(model.shipped_for_order[o, p, day,2] for day in range(1,d+1)for o in model.ORDERS) <= sum(model.released_product[p, 2,day] for day in range(1,d+1))

model.ShipmentLimit2 = Constraint(model.PRODUCTS, model.DAYS, rule=shipment_limit_rule2)


def demand_satisfaction_rule(model, o, p):
    if model.ordered_product[o, p] > 0:  # Only apply the constraint for non-zero demand
        return sum(model.shipped_for_order[o, p, d, s] 
                   for d in range(1, model.due_date[o, p] + 1) 
                   for s in model.PALLET_SIZES) == model.ordered_product[o, p]
    else:
        return Constraint.Feasible  # Skip if no demand exists

model.DemandSatisfaction = Constraint(model.ORDERS, model.PRODUCTS, rule=demand_satisfaction_rule)




def storage_constraint_rule(model,d):
    return sum(model.released_product[p,s,day]for p in model.PRODUCTS for s in model.PALLET_SIZES for day in range(1,d+1)) - sum(model.shipped_for_order[o,p,day,s] for o in model.ORDERS for p in model.PRODUCTS for day in range(1,d+1) for s in model.PALLET_SIZES )<=Q
model.StorageConstraint=Constraint(model.DAYS,rule=storage_constraint_rule)


    


def operation_shipment_rule(model,d,s):
    return sum((model.shipped_for_order[o,p,d,s]) for p in model.PRODUCTS for o in model.ORDERS) <= sum(model.operations[d,s,t] * model.capacity[t,s] for t in model.VEHICLE_TYPES)
model.operation_daily_shipment_constraint_size_one = Constraint(model.DAYS,model.PALLET_SIZES,rule=operation_shipment_rule)
''' 
def operation_vehicle_assign_rule(model,d,t):
    return model.vehicle_assigned[d,t] >= (sum(model.operations[d,s,t] for s in model.PALLET_SIZES)/3)
model.operation_vehicle_assignment = Constraint(model.DAYS,model.VEHICLE_TYPES,rule=operation_vehicle_assign_rule)
'''
def operations_rule(model, d, s, t):
    return model.operations[d, s, t] == model.owned_operations[d, s, t] + model.rented_operations[d, s, t]
model.OperationsLink = Constraint(model.DAYS, model.PALLET_SIZES, model.VEHICLE_TYPES, rule=operations_rule)

def owned_trips_limit(model, d, t):
    return sum(model.owned_operations[d, s, t] for s in model.PALLET_SIZES) <= 3 * model.vehicle_count[t]
model.OwnedTripsLimit = Constraint(model.DAYS, model.VEHICLE_TYPES, rule=owned_trips_limit)




def objective_function(model):
    owned_cost = sum(model.owned_operations[d, s, t] * model.vehicle_cost[t] 
                 for d in model.DAYS for s in model.PALLET_SIZES for t in model.VEHICLE_TYPES)
    rented_cost = sum(model.rented_operations[d, s, t] * model.vehicle_cost_rented[t] 
                  for d in model.DAYS for s in model.PALLET_SIZES for t in model.VEHICLE_TYPES)
    earliness_penalty = sum(model.earliness_penalty[o, p] * (model.due_date[o, p] - d) * model.shipped_for_order[o, p, d, s] 
                        for o in model.ORDERS for p in model.PRODUCTS 
                        for d in model.DAYS if d <= model.due_date[o, p] 
                        for s in model.PALLET_SIZES)
    return owned_cost + rented_cost + earliness_penalty
model.obj = Objective(rule=objective_function, sense=minimize)




    
    
# Calculating the CPU time
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
cpu_time = end_time - start_time



'''
# Example: Assuming 'model' is your Pyomo model
for v in model.component_objects(Var, active=True):
    print(f"\nVariable: {v.name}")
    print("Index\tValue")
    for index in v:
        print(f"{index}\t{v[index].value}")
'''  
# Print the results
print("Objective Function Value:", model.obj())
print("CPU Time:", cpu_time, "seconds")
