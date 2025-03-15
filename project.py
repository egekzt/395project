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
'''
pallet_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name = "Pallets" ) 
vehicle_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name= "Vehicles")  
order_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name = "Orders")
'''
'''
pallet_data = pd.read_excel("ProjectPart1-Scenario2.xlsx",sheet_name = "Pallets" ) 
vehicle_data = pd.read_excel("ProjectPart1-Scenario2.xlsx",sheet_name= "Vehicles")  
order_data = pd.read_excel("ProjectPart1-Scenario2.xlsx",sheet_name = "Orders")
'''


parameters_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name = "Parameters", engine="openpyxl")
    
# Convert to dictionary
parameters = dict(zip(parameters_data["Parameter"], parameters_data["Value"]))
    
# Assign values
T = parameters.get("Planning Horizon (T)", None)
tripcount = parameters.get("Max. trips per period", None)
Q = parameters.get("Max. pallets in area", None)
    


pallet_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name = "Pallets" ) 
vehicle_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name= "Vehicles")  
order_data = pd.read_excel("ProjectPart1-Scenario1.xlsx",sheet_name = "Orders")





'''here since for all datasets we are given, these are set values which
do not change for any scenario we preferred hardcoding them but of course
we could also read just as we did the rest'''



#given our q and capacity values never change within the scope of the given datasets, we are hardcoding them
vehicle_capacity ={
        (1,1): 22,
        (1,2) : 33,
        (2,1): 12,
        (2,2) : 18,
        (3,1): 6,
        (3,2) : 8,
}

#horizon
model.DAYS = Set(initialize=range(1,T+1)) 
#need to be decided based on the type of datasets i will be given both days and products
model.PRODUCTS = Set(initialize=pallet_data['Product Type'].unique()) 
model.PALLETS = Set(initialize=pallet_data["Pallet ID"].unique())
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

model.shipped_pallet=Var(model.PALLETS,model.DAYS,domain=Binary)
#if pallet consists type 1 items or not,likewise down below ps2 same which applies to type2 pallets
model.pallet_size1 = Param(model.PALLETS, 
                           initialize={
                               row['Pallet ID']: 1 if row['Pallet Size'] == 1 else 0 
                               for _, row in pallet_data.iterrows()
                           }, 
                           default=0)

model.pallet_size2 = Param(model.PALLETS, 
                           initialize={
                               row['Pallet ID']: 1 if row['Pallet Size'] == 2 else 0 
                               for _, row in pallet_data.iterrows()
                           }, 
                           default=0)

model.pallet_product = Param(model.PALLETS, initialize={
    row['Pallet ID']: row['Product Type'] for _, row in pallet_data.iterrows()
})

model.pallet_amount = Param(model.PALLETS, initialize={
    row['Pallet ID']: row['Amount'] for _, row in pallet_data.iterrows()
})


model.released_pallet = Param(model.PALLETS, model.DAYS, mutable=True, default=0)
for _, row in pallet_data.iterrows():
    p = row['Pallet ID']
    d = row['Release Day']
    model.released_pallet[p, d] = 1  # Binary: 1 if released on day d


'''
#to be able to update available product we need to be able to tell how much releases when
model.released_product = Param(model.PRODUCTS,model.PALLET_SIZES, model.DAYS, domain=NonNegativeIntegers,default = 0,mutable=True)  
'''




#the order set 
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

# Define binary variable delivered_early for each order and product
model.delivered_early = Var(model.ORDERS, model.PRODUCTS, domain=Binary, initialize=0)


'''since here on our model we decide how many operations will suffice
to be able to lift the amount of product we need to ship and the cost
calculation we need to keep operations and then operations owned and rented 
seperately, since we are given owned cars can only do 3 trips, after that
a rented car will do a trip for a fixed cost anyways so we just need to be able
to deduce from operation count rented and owned seperately for price calculation'''
model.owned_operations=Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers,initialize=0)
model.rented_operations=Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers,initialize=0)
model.operations = Var(model.DAYS,model.PALLET_SIZES,model.VEHICLE_TYPES,domain=NonNegativeIntegers,initialize=0) 
 


'''to be able to keep track of early delivers its better to keep the deliveries
on each day for which order that they belong to so we can track it'''
model.shipped_for_order=Var(model.ORDERS,model.PRODUCTS,model.DAYS,model.PALLET_SIZES,initialize=0,domain=NonNegativeIntegers)




#filling the necessary variables from the table vehicle_data
for index,row in vehicle_data.iterrows():
    vehicle_type = row["Vehicle Type"]
    model.vehicle_cost[vehicle_type] = row["Fixed Cost (c_k)"]
    model.vehicle_cost_rented[vehicle_type] = row["Variable Cost (c'_k)"]

''' First, group the data by 'Vehicle Type' and sum the 'Num of vehicles'
since we are not given them but rather they are seperate rows on the data
'''
vehicle_data_grouped = vehicle_data.groupby('Vehicle Type').size().reset_index(name='Num of vehicles')

# Now iterate over the grouped rows
for index, row in vehicle_data_grouped.iterrows():
    amount = row['Num of vehicles']
    vehicle_type = row['Vehicle Type']
    model.vehicle_count[vehicle_type] = amount
    


#making sure we dont release more than once
def released_once_rule(model, p):
    return sum(model.released_pallet[p, d] for d in model.DAYS) == 1

model.released_once_constraint = Constraint(model.PALLETS, rule=released_once_rule)
#making sure we dont ship more than once
def shipped_once_rule(model, p):
    return sum(model.shipped_pallet[p, d] for d in model.DAYS) == 1

model.shipped_once_constraint = Constraint(model.PALLETS, rule=shipped_once_rule)

#we dont ship before release
def release_before_shipped_constraint(model,pallet,day):
    return sum(model.released_pallet[pallet, d] for d in range(1,day+1)) >= model.shipped_pallet[pallet, day]
model.shipped_release_constraint=Constraint(model.PALLETS,model.DAYS,rule=release_before_shipped_constraint)
'''
making sure we dont have any product that are not unassigned, that are allready shipped, and allocate them to 
best orders that there are for them
'''
def shipment_limit_rule(model, p, d):
    return sum(
        model.shipped_pallet[pallet, day] * model.pallet_size1[pallet] * model.pallet_amount[pallet]
        for pallet in model.PALLETS 
        if model.pallet_product[pallet] == p
        for day in model.DAYS if day <= d
    ) == sum(
        model.shipped_for_order[o, p, day, 1]
        for o in model.ORDERS 
        for day in model.DAYS if day <= d
    )
model.ShipmentLimit = Constraint(model.PRODUCTS, model.DAYS, rule=shipment_limit_rule)

def shipment_limit_rule2(model, p, d):
    return sum(
        model.shipped_pallet[pallet, day] * model.pallet_size2[pallet] * model.pallet_amount[pallet]
        for pallet in model.PALLETS 
        if model.pallet_product[pallet] == p
        for day in model.DAYS if day <= d
    ) == sum(
        model.shipped_for_order[o, p, day, 2]
        for o in model.ORDERS 
        for day in model.DAYS if day <= d
    )
model.ShipmentLimit2 = Constraint(model.PRODUCTS, model.DAYS, rule=shipment_limit_rule2)


#we have to check if theres any early delivery for any product
def early_delivery_indicator_rule(model, o, p):
    # Sum all shipments for order o and product p delivered before the due date.
    early_shipments = sum(
        model.shipped_for_order[o, p, d, s]
        for d in model.DAYS if d < model.due_date[o, p]
        for s in model.PALLET_SIZES
    )
    # Big M: using ordered demand as an upper bound.
    M = model.ordered_product[o, p]
    # If any early shipments occur, early_shipments >= 1 forcing delivered_early to be 1.
    return early_shipments <= M * model.delivered_early[o, p]

model.early_delivery_indicator = Constraint(model.ORDERS, model.PRODUCTS, rule=early_delivery_indicator_rule)


'''now here we have to make sure every demand is satisfied by their due date
and to do that all we have to do, is sum the delivered shipment until due date
of all orders and force them to equal'''
def demand_satisfaction_rule(model, o, p):
    if model.ordered_product[o, p] > 0:  # Only apply the constraint for non-zero demand
        return sum(model.shipped_for_order[o, p, d, s] 
                   for d in range(1, model.due_date[o, p] + 1) 
                   for s in model.PALLET_SIZES) == model.ordered_product[o, p]
    else:
        return Constraint.Feasible  # Skip if no demand exists

model.DemandSatisfaction = Constraint(model.ORDERS, model.PRODUCTS, rule=demand_satisfaction_rule)



'''since released-shipped till the day we are at cumulatively gives us whatever
there is left in the production facility we utilize this and compare that to given Q
we have to make sure its less than Q'''
def storage_constraint_rule(model, d):
    return sum(model.released_pallet[p, day] - model.shipped_pallet[p, day]
               for p in model.PALLETS for day in model.DAYS if day <= d) <= Q
model.StorageConstraint = Constraint(model.DAYS, rule=storage_constraint_rule)
    


'''here as i have explained before we decide that the amount of sufficient operations
it has to be able to carry the amount of shipment we plan to make and from there
we can decide the amount of rented and owned operation counts'''
def operation_shipment_ruleONE(model, d):
    return sum(model.shipped_pallet[p, d] * model.pallet_size1[p] for p in model.PALLETS) <= sum(model.operations[d, 1, t] * model.capacity[t, 1] for t in model.VEHICLE_TYPES)

model.operation_daily_shipment_constraint_size_one = Constraint(model.DAYS, rule=operation_shipment_ruleONE)

def operation_shipment_ruleTWO(model, d):
    return sum(model.shipped_pallet[p, d] * model.pallet_size2[p] for p in model.PALLETS) <= sum(model.operations[d, 2, t] * model.capacity[t, 2] for t in model.VEHICLE_TYPES)

model.operation_daily_shipment_constraint_size_two = Constraint(model.DAYS, rule=operation_shipment_ruleTWO)


def operations_rule(model, d, s, t):
    return model.operations[d, s, t] == model.owned_operations[d, s, t] + model.rented_operations[d, s, t]
model.OperationsLink = Constraint(model.DAYS, model.PALLET_SIZES, model.VEHICLE_TYPES, rule=operations_rule)

def owned_trips_limit(model, d, t):
    return sum(model.owned_operations[d, s, t] for s in model.PALLET_SIZES) <= tripcount * model.vehicle_count[t]
model.OwnedTripsLimit = Constraint(model.DAYS, model.VEHICLE_TYPES, rule=owned_trips_limit)



'''owned cost would be the amount of owned operations*owned car cost per trip
rented cost likewise for rented cars
and earliness penalty is the shipment amount * earliness penalty * days theres left for due date
'''
model.shipment_day = Var(model.ORDERS, model.PRODUCTS, model.DAYS, domain=Binary)

# Link shipment_day to shipped_for_order
def link_shipment_day_rule(model, o, p, d):
    if d >= model.due_date[o, p]:
        return Constraint.Skip  # Only consider days before due date
    total_shipped = sum(model.shipped_for_order[o, p, d, s] for s in model.PALLET_SIZES)
    M = model.ordered_product[o, p]  # Use demand as big-M
    return total_shipped <= M * model.shipment_day[o, p, d]

model.LinkShipmentDay = Constraint(
    model.ORDERS, model.PRODUCTS, model.DAYS, rule=link_shipment_day_rule
)



# Update objective to use shipment_day for earliness penalty
def objective_function(model):
    owned_cost = sum(
        model.owned_operations[d, s, t] * model.vehicle_cost[t]
        for d in model.DAYS for s in model.PALLET_SIZES for t in model.VEHICLE_TYPES
    )
    rented_cost = sum(
        model.rented_operations[d, s, t] * model.vehicle_cost_rented[t]
        for d in model.DAYS for s in model.PALLET_SIZES for t in model.VEHICLE_TYPES
    )
    # Earliness penalty: days early × penalty (no quantity multiplier)
    earliness_penalty_cost = sum(
        model.earliness_penalty[o, p] * (model.due_date[o, p] - d) * model.shipment_day[o, p, d]
        for o in model.ORDERS for p in model.PRODUCTS
        for d in model.DAYS if d < model.due_date[o, p]
    )
    return owned_cost + rented_cost + earliness_penalty_cost

model.obj = Objective(rule=objective_function, sense=minimize)







    
    
# Calculating the CPU time
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
cpu_time = end_time - start_time







# Print the results
print("Objective Function Value:", model.obj())
print("CPU Time:", cpu_time, "seconds")
