"""Conservative pre-rental estimate. This module never creates paid resources."""
import json
import math
from pathlib import Path


def estimate(path,gpu_hourly,storage_monthly_gb,disk_gb,hours,transfer=0,margin=1.3):
    values=[gpu_hourly,storage_monthly_gb,disk_gb,hours,transfer,margin]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values) or hours==0 or disk_gb==0 or margin<1:
        raise ValueError("Costs must be finite nonnegative; runtime/disk positive; margin >= 1")
    ledger=json.loads(Path(path).read_text())
    if ledger['active_instance_ids']:
        raise ValueError("Reconcile/destroy task-active instances before planning another rental")
    consumed=max(ledger['estimated_spend'],ledger['actual_spend'] or 0)
    remaining=ledger['initial_budget']-consumed
    storage=storage_monthly_gb*disk_gb*hours/720
    gpu=gpu_hourly*hours
    conservative=(gpu+storage+transfer)*margin
    usable=remaining-ledger.get('recovery_reserve',2)
    if conservative>usable:
        raise ValueError("Estimate exceeds available experimentation budget after recovery reserve")
    return {'gpu_hourly_usd':gpu_hourly,'runtime_hours':hours,'disk_gb':disk_gb,'disk_monthly_per_gb_usd':storage_monthly_gb,'gpu_cost_usd':gpu,'disk_cost_usd':storage,'transfer_cost_usd':transfer,'margin_multiplier':margin,'approximate_total_usd':conservative,'remaining_estimated_after_plan_usd':remaining-conservative,'reserve_usd':ledger.get('recovery_reserve',2),'rental_created':False,'note':'Estimate only; requires verified offer/account balance and real model/dataset readiness before creation. Includes 720-hour month approximation; verify provider billing.'}
