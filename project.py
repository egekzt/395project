# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:39:20 2025

@author: egeki
"""
import time
import pyomo.environ as pyomo
import pandas as pd
from pyomo.environ import *

model = pyomo.ConcreteModel()
solver = SolverFactory('gurobi')
results = solver.solve(model,tee=True)
#file names needs to be implemented after datasets are given out
pallet_data = pd.read_csv("pallet_data.csv") 
vehicle_data = pd.read_csv("vehicle_data.csv")
order_data = pd.read_csv("order_data.csv")

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
#to be able to update available product we need to be able to tell how much releases when
model.released_product = Var(model.PRODUCTS,model.PALLET_SIZES, model.DAYS, domain=NonNegativeIntegers)  
#for available product to make decisions to send & to make sure at the end of the day
model.available_product = Var(model.PRODUCTS,model.PALLET_SIZES, model.DAYS, domain=NonNegativeIntegers) 
               
model.ORDER_PRODUCT_PAIRS = Set(initialize={(row['Order ID'], row['Product Type']) 
                                            for _, row in order_data.iterrows()})

# to be able to compare the demand with shipped amount to make sure we dont underdeliver
model.ordered_product = Param(model.ORDER_PRODUCT_PAIRS, model.DAYS, 
                              initialize={(row['Order ID'], row['Product Type'], row['Due date']): row['Demand Amount'] 
                                          for _, row in order_data.iterrows()}, 
                              default=0)

# to keep the due dates to be able to make the necessary comparation
model.due_date = Param(model.ORDER_PRODUCT_PAIRS, 
                       initialize={(row['Order ID'], row['Product Type']): row['Due date'] 
                                   for _, row in order_data.iterrows()})

# for the amount that is unnecessary but delivered we have to pay earliness penalty
model.earliness_penalty = Param(model.ORDER_PRODUCT_PAIRS, 
                                initialize={(row['Order ID'], row['Product Type']): row['Earliness penalty'] 
                                            for _, row in order_data.iterrows()}, default=0)


                                                                             

#given on homework text we are on a range of atmost 7 so i initialize the range as 1-7 but could be read from csv aswell
model.DAYS = Set(initialize=range(1,8)) 
#need to be decided based on the type of datasets i will be given both days and products
model.PRODUCTS = Set(initialize=pallet_data['Product'].unique()) 
#only 3 types of vehicles are declared so we can hardcode
model.VEHICLE_TYPES = Set(initialize = [1,2,3]) 
#where 1 corresponds to 100x120 and 2 is 80x120 same as vehicle types we can hardcode since wont be more sizes
model.PALLET_SIZES = Set(initialize = [1,2]) 
#to decide the amount of product type and size shipped each day
model.shipped_product = Var(model.PRODUCTS,model.PALLET_SIZES,model.DAYS,domain=NonNegativeIntegers) 
#as named its for getting the capacity for given vehicle type with given size
model.capacity = Param(model.VEHICLE_TYPES, model.PALLET_SIZES,initialize=vehicle_capacity) 
#tracking cost of vehicles seperately
model.vehicle_cost = Param(model.VEHICLE_TYPES,initialize={})
model.vehicle_cost_rented =Param(model.VEHICLE_TYPES,initialize={})
#count of owned vehicles for our company so we can decide after what point they count as rental
model.vehicle_count =Param(model.VEHICLE_TYPES,initiaize={})
#to hold operation count for each vehicle and day 
model.operations = Var(model.VEHICLE_TYPES,model.DAYS,domain=NonNegativeIntegers) 
#the amount of vehicle assigned for the specific day, type and size 
model.vehicle_assigned = Var(model.DAYS,model.VEHICLE_TYPES,model.PALLET_SIZES)
#the amount of vehicle rented
model.vehicle_rented = Var(model.DAYS, model.VEHICLE_TYPES, model.PALLET_SIZES, domain=NonNegativeIntegers)

#filling the necessary variables from the table vehicle_data
for index,row in vehicle_data.iterrows():
    vehicle_type = vehicle_type_mapping[row['Vehicle']]
    model.vehicle_cost[vehicle_type] = row['Fixed cost']
    model.vehicle_cost_rented[vehicle_type] = row['Variable cost']

for index, row in vehicle_data.iterrows():
    amount = row['Num. of vehicles ']
    vehicle_type_temp = row['Vehicle']
    vehicle_type = vehicle_type_mapping.get(vehicle_type_temp,None) #since ive used values for vehicle types but table has names i should be using a mapper to reach ints
    
    if vehicle_type is not None:
        model.vehicle_count[vehicle_type] = amount
    else:
        print("Error") #normally it should never turn None technically but to be able to catch an unexpected case i just error handle simply

#grabbing the data from pallet_data table
for index, row in pallet_data.iterrows(): #here from the table we calculate released product count for each day for every size and product type
    product = row['Product']
    release_day = row['Release day']
    amount = row['Product amount']
    size_type = row['Size type']
    model.released_product[product,size_type,release_day] += amount

#to be able to keep track of the available_product each day we sum the released and then extract the shipped amount from that, this gives us the available amount
#which we need to be able to tell how much we have that we can able to ship + the Q constraint where for each day we have to make sure avaiable_product doesnt exceeds Q
for product in model.PRODUCTS:
    for size_type in model.PALLET_SIZES:
        for day in model.DAYS:
            model.available_product[product, size_type, day] = (
                sum(model.released_product[product, size_type, d] for d in range(1, day+1)) 
                - sum(model.shipped_product[product, size_type, d] for d in range(1, day+1))
)
#to make sure we dont exceed the limit for 3 operations for each vehicle on a day
def operation_limit_rule(model,k,d): 
    return model.operations[k,d] <= 3*model.vehicle_count[k]
model.operation_limit = Constraint(model.VEHICLE_TYPES,model.DAYS,rule=operation_limit_rule)

#to make sure amount of pallets left available does not exceed Q
def waiting_area_limit_rule(model,d): 
    return sum(model.available_product[i,d] for i in model.PRODUCTS) <= Q #q will be given
model.waiting_area = Constraint(model.PRODUCTS,rule=waiting_area_limit_rule)

#to make sure no order is completed after the date
def operation_lateness_rule(model,o,p):
    return sum(model.shipped_product[p,s,d] for s in model.PALLET_SIZES for d in range(1,model.due_date[o,p]+1)) >= sum(model.ordered_product[o,p,d] for d in range(1,model.due_date[o,p]+1))
model.lateness = Constraint(model.ORDER_PRODUCT_PAIRS, rule=operation_lateness_rule)  

#our two constraints for corresponding size, for the assigned vehicle counts we make sure we can carry enough
def operation_shipment_rule_one(model,d):
     shipped_size_one = sum(model.shipped_product[p,1,d] for p in model.PRODUCTS)
     available_capacity_to_transport_size_one = sum(model.vehicle_assigned[d,t,1]*model.vehicle_capacity[t,1] for t in model.VEHICLE_TYPES)
     return shipped_size_one <= available_capacity_to_transport_size_one
model.operation_daily_shipment_constraint_size_one = Constraint(model.DAYS,rule=operation_shipment_rule_one)

def operation_shipment_rule_two(model,d):
     shipped_size_two = sum(model.shipped_product[p,2,d] for p in model.PRODUCTS)
     available_capacity_to_transport_size_two = sum(model.vehicle_assigned[d,t,2]*model.vehicle_capacity[t,2] for t in model.VEHICLE_TYPES)
     return  shipped_size_two <= available_capacity_to_transport_size_two
model.operation_daily_shipment_constraint_size_two = Constraint(model.DAYS,rule=operation_shipment_rule_two)

#since im using a formula which i pick the amount of vehicle assigned to keep track for each day, based on type we can just assigned-owned to get the rented value
#on the obj function both variables needs to be multiplied by the corresponding values to their own so we have to keep them sepearate
def rented_vehicle_rule(model, d, v, p):
    return model.vehicle_rented[d, v, p] >= model.vehicle_assigned[d, v, p] - model.vehicle_count[v]

model.rented_vehicle_constraint = Constraint(model.DAYS, model.VEHICLE_TYPES, model.PALLET_SIZES, rule=rented_vehicle_rule)

def objective_function(model):
    rental_car_cost = sum(model.vehicle_cost_rented[t] * model.vehicle_rented[d,t,s] for t in model.VEHICLE_TYPES for d in model.DAYS for s in model.PALLET_SIZES)
    owned_car_cost = sum((model.vehicle_assigned[d,t,s] - model.vehicle_rented[d,t,s])*model.vehicle_cost[t] for t in model.VEHICLE_TYPES for d in model.DAYS for s in model.PALLET_SIZES)
    #to calculate earliness penalty we have to ask, how do we decide which penalty we pick, the closest to our deadline ? or the one with lowest cost
    return owned_car_cost + rental_car_cost

model.obj = Objective(rule=objective_function, sense=minimize) 
objective_value = model.objective_function.expr()
    
    
#calculating the cpu time
start_time = time.time()
solver.solve(model)
end_time = time.time()
cpu_time = end_time - start_time

gap = solver.results.problem.upper_bound - solver.results.problem.lower_bound
if solver.results.problem.upper_bound != 0:
    gap_percentage = (gap / solver.results.problem.upper_bound) * 100
else:
    gap_percentage = 0  #error handle we dont wanna divide by 0 and throw exceptions

print("Objective Function Value:", objective_value)
print("CPU Time:", cpu_time, "seconds")
print("Gap Percentage:", gap_percentage if gap_percentage is not None else "N/A")