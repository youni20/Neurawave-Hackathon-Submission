

person_id,timestamp,stress_intensity,stress_duration,stress_letdown,menstruation,delivery,combined_pill,sleep_deficit,sleep_fragmentation,red_wine,caffeine,cheddar,bluemold_cheese,parmesan,salami,bacon,sausage,sweeteners,screen_time,bright_light,loud_noise,pressure_change,temp_change,precipitation_change,sun_cloud_change,extreme_temp,missed_meal,low_blood_sugar,medicine_nitrates,medicine_vasodilators,medicine_hormonal_pills,medicine_hormonal_patch,medicine_estrogen_substitution,medicine_painkillers,medicine_NSAID,medicine_triptan,medicine_combined_painkillers,medicine_SSRI_antidepressants,migraine_time,migraine_intensity
1,2025-05-01T08:00:00,7,120,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,2,0,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,2025-05-01T10:00:00,6
2,2025-05-01T09:00:00,8,90,0,1,0,1,0,0,0,0,1,0,1,0,1,1,0,3,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,2025-05-01T12:00:00,7
3,2025-05-01T10:00:00,5,150,1,0,0,0,1,1,1,0,0,0,1,0,1,0,1,1,2,1,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,2025-05-01T14:00:00,4


person_id: Unique UUID for the person.

timestamp: Time of measurement (e.g., when data was collected).

stress_intensity: Intensity of stress (1-10).

stress_duration: Duration of stress (in minutes).

stress_letdown: Whether a sudden decrease in stress occurred (1 = yes, 0 = no).

menstruation: Whether the person is menstruating (1 = yes, 0 = no).

delivery: Whether the person recently gave birth (1 = yes, 0 = no).

combined_pill: Whether the person uses combined birth control pills (1 = yes, 0 = no).

sleep_deficit: Whether the person has sleep deficit (1 = yes, 0 = no).

sleep_fragmentation: Whether the person has sleep fragmentation (1 = yes, 0 = no).

red_wine: Whether the person has consumed red wine (1 = yes, 0 = no).

caffeine: Whether the person has consumed >200 mg caffeine or sudden caffeine withdrawal (1 = yes, 0 = no).

cheese_meats: Whether the person has eaten cheddar, blue cheese or parmesan, as well as salami, bacon or sausage (1 = yes, 0 = no).

sweeteners: Whether the person has used sweeteners (1 = yes, 0 = no).

screen_time: Whether the person has had a lot of screen time (1 = yes, 0 = no).

bright_light: Whether the person has been exposed to bright light (1 = yes, 0 = no).

loud_noise: Whether the person has been exposed to loud sounds (>85 dB) (1 = yes, 0 = no).

pressure_change: Whether the person has experienced rapid barometric pressure change (1 = yes, 0 = no).

temp_change: Whether the person has experienced rapid temperature change (1 = yes, 0 = no).

precipitation_change: Whether the person has experienced rapid change in precipitation (1 = yes, 0 = no).

sun_cloud_change: Whether the person has experienced rapid change between sun and clouds (1 = yes, 0 = no).

extreme_temp: Whether the person has been exposed to extreme temperature (cold or hot) (1 = yes, 0 = no).

missed_meal: Whether the person has missed a meal (1 = yes, 0 = no).

low_blood_sugar: Whether the person has had low blood sugar (1 = yes, 0 = no).

medications: Whether the person has used medications (a list, e.g. "nitrates, SSRI").

migraine_time: Time of migraine (if applicable).

migraine_intensity: Intensity of migraine (1-10).

cheddar: Whether the person has eaten cheddar (1 = yes, 0 = no).

bluemold_cheese: Whether the person has eaten blue cheese (1 = yes, 0 = no).

parmesan: Whether the person has eaten parmesan (1 = yes, 0 = no).

salami: Whether the person has eaten salami (1 = yes, 0 = no).

bacon: Whether the person has eaten bacon (1 = yes, 0 = no).

sausage: Whether the person has eaten sausage (1 = yes, 0 = no).

nitrates: Whether the person has used nitrates (1 = yes, 0 = no).

vasodilators: Whether the person has used vasodilators (1 = yes, 0 = no).

hormonal_pills: Whether the person has used combined birth control pills (1 = yes, 0 = no).

hormonal_patch: Whether the person has used hormonal patches (1 = yes, 0 = no).

estrogen_substitution: Whether the person has used estrogen substitution (1 = yes, 0 = no).

painkillers: Whether the person has used painkillers (1 = yes, 0 = no).

NSAID: Whether the person has used NSAIDs (1 = yes, 0 = no).

triptan: Whether the person has used triptans (1 = yes, 0 = no).

combined_painkillers: Whether the person has used combined painkillers (1 = yes, 0 = no).

SSRI_antidepressants: Whether the person has used SSRI antidepressants (1 = yes, 0 = no).