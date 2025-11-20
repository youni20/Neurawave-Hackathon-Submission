# Synthetic data overview

## Example `health_data_[persons]_[days].csv` file content:

```csv
person_id,migraine_days_per_month,timestamp,stress_intensity,menstruation,delivery,sleep_duration,sleep_deficit,missed_meal,timestamp_dt,p_stress,p_hormones,p_sleep,p_weather,p_meals,migraine_probability,migraine
bec5482d-375f-4cb1-99ed-898876e8db34,2,2024-01-01T18:38:05.534198574,0,False,False,318.8389309205603,True,False,2024-01-01 18:38:05.534198,0.0,0.0,0.0,0.0,0.0,0.0,False
bec5482d-375f-4cb1-99ed-898876e8db34,2,2024-01-02T08:15:36.842608093,2,False,False,319.5269448600329,True,False,2024-01-02 08:15:36.842608,0.0,0.0,0.0,0.0,0.0,0.0,False
bec5482d-375f-4cb1-99ed-898876e8db34,2,2024-01-03T20:01:13.609812080,0,False,False,318.3592139453124,True,False,2024-01-03 20:01:13.609812,0.0,0.0,0.0,0.0,0.0,0.0,False
...
```

## Column descriptions:

**Person-specific data:**

- person_id: Unique UUID for the person.
- migraine_days_per_month: Average number of expected migraine days per month for this person.

**Day-specific data:**

- timestamp: Time of measurement (e.g., when data was collected).
- timestamp_dt: Datetime object of the timestamp.
- stress_intensity: Intensity of stress (1-10).
- menstruation: Whether the person is menstruating (1 = yes, 0 = no).
- delivery: Whether the person recently gave birth (1 = yes, 0 = no).
- sleep_duration: Duration of sleep in minutes.
- sleep_deficit: Whether the person has sleep deficit (1 = yes, 0 = no).
- missed_meal: Whether the person has missed a meal (1 = yes, 0 = no).

**Probability contributions:**

- p_stress: Probability contribution from stress factors.
- p_hormones: Probability contribution from hormonal factors.
- p_sleep: Probability contribution from sleep factors.
- p_weather: Probability contribution from weather factors.
- p_meals: Probability contribution from meal factors.
- migraine_probability: Overall probability of migraine occurrence.

**Outcome:**

- migraine: Whether a migraine occurred (1 = yes, 0 = no).

