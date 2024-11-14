# **Scania challenge 2016**

# **Data Science applied to reduce maintenance costs**

## **What**

Reduce maintenance costs for air pressure systems in trucks.
The maintenance cost of a specific system is shown below in dollars:

- If a truck is sent for maintenance, but it does not show any defect in this system, around **$10** will be charged for the time spent during the inspection by the specialized team.

- If a truck is sent for maintenance and it is defective in this system, **$25** will be charged to perform the preventive repair service.

- If a truck with defects in the air system is not sent directly for maintenance, the company pays **$500** to carry out corrective maintenance of the same, considering the labor, replacement of parts and other possible inconveniences (truck broke down in the middle of the track for example).

## **Why**

A company maintains an average number of trucks in its fleet to deliver across the country, but in the last 3 years it has been noticing a large increase in the expenses related to the maintenance of the air system of its vehicles, even though it has been keeping the size of its fleet relatively constant.

The main goal of this project was to estimate the cost of maintenance for the following year and reduce the total spent.

## **How**

The technical team informed that all information regarding the air system of the paths will be made available at the train dataset _air_system_previous_years.csv_ where they created a column in the database called **class**.

*   Class "pos" would be those trucks that had defects in the air system

*   Class "neg" would be those trucks that had a defect in any system other than the air system

Supervised machine learning techniques were used to solve the problem and the **Random Forest, XGBoost, Multi-layer Perceptron and AdaBoosting** models were applied.

As test dataset were used the _air_system_present_year.csv_.

The final model used to predict the classification was **Random Forest with hyper parameters.**

## **Savings**

![historical_cost](https://i.ibb.co/P6gHV0h/historical-cost.png)

![pct_variations](https://i.ibb.co/d5HXr5F/pct-cost-variation.png)