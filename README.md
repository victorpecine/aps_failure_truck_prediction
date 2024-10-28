# **Data Science applied to maintenance planning optimization**

### The answers for this challenge can be found at _report_air_system_failure.pdf_

### The code with pipelines for EDA and machine learning models are at _pipeline.ipynb_

## **Situation**
A new data science consulting company was hired to solve and improve the maintenance planning of an outsourced transport company. The company maintains an average number of trucks in its fleet to deliver across the country, but in the last 3 years it has been noticing a large increase in the expenses related to the maintenance of the air system of its vehicles, even though it has been keeping the size of its fleet relatively constant. The maintenance cost of this specific system is shown below in dollars:

Your objective as a consultant is to decrease the maintenance costs of this particular system. Maintenance costs for the air system may vary depending on the actual condition of the truck.

- If a truck is sent for maintenance, but it does not show any defect in this system, around $10 will be charged for the time spent during the inspection by the specialized team.

- If a truck is sent for maintenance and it is defective in this system, $25 will be charged to perform the preventive repair service.

- If a truck with defects in the air system is not sent directly for maintenance, the company pays $500 to carry out corrective maintenance of the same, considering the labor, replacement of parts and other possible inconveniences (truck broke down in the middle of the track for example).

During the alignment meeting with those responsible for the project and the company's IT team, some information was given to you:

- The technical team informed you that all information regarding the air system of the paths will be made available to you, but for bureaucratic reasons regarding company contracts, all columns had to be encoded.

- The technical team also informed you that given the company's recent digitization, some information may be missing from the database sent to you.

Finally, the technical team informed you that the source of information comes from the company's maintenance sector, where they created a column in the database called **class**: "pos" would be those trucks that had defects in the air system and "neg" would be those trucks that had a defect in any system other than the air system.

Those responsible for the project are very excited about the initiative and, when asking for a technical proof of concept, they have put forth as main requirements:

- Can we reduce our expenses with this type of maintenance using AI techniques?

- Can you present to me the main factors that point to a possible failure in this system?


These points, according to them, are important to convince the executive board to embrace the cause and apply it to other maintenance systems during the year 2022.

## **About the database**

Two files will be sent to you:

- _air_system_previous_years.csv_: File containing all information from the maintenance sector for years prior to 2022 with 178 columns.

- _air_system_present_year.csv_: File containing all information from the maintenance sector in this year.

- Any missing value in the database is denoted by _na_.

The final results that will be presented to the executive board need to be evaluated against _air_system_present_year.csv_.