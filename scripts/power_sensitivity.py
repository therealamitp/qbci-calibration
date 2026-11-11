"""Normal-approximation paired-effect power, assumptions explicit."""
import json
from scipy.stats import norm

rows=[]
for n in (9,18,54):
    for sd in (0.05,0.08,0.10):
        se=sd/n**0.5
        delta=0.03
        z=norm.ppf(0.975)
        power=float(norm.sf(z-delta/se)+norm.cdf(-z-delta/se))
        rows.append({"n_independent_participants":n,"paired_effect_sd":sd,
                     "true_effect":delta,"two_sided_alpha":0.05,"approximate_power":power,
                     "effect_for_80_percent_power":float((z+norm.ppf(0.8))*se)})
print(json.dumps({"scope":"PLANNING ASSUMPTIONS, NOT ESTIMATED EEG POWER", "scenarios":rows},indent=2))
