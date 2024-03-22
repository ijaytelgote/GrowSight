import random
import pandas as pd
from faker import Faker

fake = Faker()

class DataGenerator:
    @staticmethod
    def generate_fake_data(rows=100):
        data = []

        # Generating date ranges outside the loop
        last_email_sent_dates = pd.date_range(start='2023-01-01', periods=rows, freq='M').strftime('%B %Y')
        last_interaction_dates = pd.date_range(start='2020-10-01', periods=rows, freq='M').strftime('%B %Y')
        last_meeting_dates = pd.date_range(start='2022-12-01', periods=rows, freq='M').strftime('%B %Y')
        last_phone_call_dates = pd.date_range(start='2003-01-06', periods=rows, freq='M').strftime('%B %Y')

        for i in range(rows):
            row = {
                'Timestamp': fake.date_time_between(start_date='-30d', end_date='now'),
                'comment': fake.text(),
                "Monthly Revenue": random.randint(1000, 10000),
                "Opportunity Amount": random.randint(10000, 100000),
                "Support Tickets Open": random.randint(0, 10),
                "Support Tickets Closed": random.randint(0, 10),
                "Lead Score": random.randint(0, 100),
                "Age": random.randint(18, 90),
                "Size": random.uniform(5, 30),
                "Continent": random.choice(["Asia", "Europe", "Africa", "Americas"]),
                "Contract Type": random.choice(["One-time", "Recurring"]),
                "Gender": random.choice(["Male", "Female"]),
                "Lead Status": random.choice(["Qualified", "Unqualified", "Contacted", "Not Contacted"]),
                "Country": fake.country(),
                "Population": random.randint(1000000, 1000000000),
                "Area (sq km)": random.randint(100000, 10000000),
                "GDP (USD)": random.randint(1000000, 10000000000),
                'Last Email Sent Date': last_email_sent_dates[i],
                'Last Interaction Date': last_interaction_dates[i],
                'Last Meeting Date': last_meeting_dates[i],
                'Last Phone Call Date': last_phone_call_dates[i],
                'Probability of Close': random.randint(0, 100)
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df
DataGenerator.generate_fake_data()
