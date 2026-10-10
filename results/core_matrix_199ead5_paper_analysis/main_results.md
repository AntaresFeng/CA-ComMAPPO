# Main final-test results

Training seed is the statistical unit. Values are mean ± sample std; 
fixed static policies are reported as point estimates.

| Algorithm | Seeds | arrival_rate | collision_rate | mean_episode_reward | mean_throughput | mean_waiting_time | mean_travel_time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention-MAPPO | 3 | 0.7533 ± 0.1274 | 0.0133 ± 0.0153 | 9.5825 ± 0.2893 | 0.2041 ± 0.0099 | 3.0952 ± 0.3061 | 18.4007 ± 0.3915 |
| CA-ComMAPPO | 3 | 0.8267 ± 0.2003 | 0.0333 ± 0.0231 | 9.3217 ± 1.1844 | 0.2107 ± 0.0222 | 2.8080 ± 0.7236 | 18.0940 ± 0.9184 |
| CA no-prior | 3 | 0.3933 ± 0.0896 | 0.1200 ± 0.0781 | 5.8518 ± 1.8924 | 0.1713 ± 0.0024 | 3.9878 ± 0.4534 | 18.2600 ± 0.9834 |
| Idle-only | 1 | 0.1700 | 0.8300 | -16.1293 | 0.0716 | 0.0000 | 5.6120 |
| MAPPO | 3 | 0.3600 ± 0.1212 | 0.1233 ± 0.1793 | 5.1975 ± 5.1739 | 0.1642 ± 0.0265 | 3.7540 ± 0.7936 | 17.2713 ± 2.5656 |
| Random | 1 | 0.1000 | 0.7500 | -12.4574 | 0.0522 | 0.6925 | 11.2360 |
