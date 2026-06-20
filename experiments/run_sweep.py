from src.data_generator import generate_data
from src.shifts import group_shift, covariate_shift, label_shift


def run_sweep():
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [1.0, 2.0, 3.0, 4.0]
    betas = [0.0, 0.05, 0.1, 0.15, 0.2]

    
    print("Group Shift:")
    for severity in alphas:
        X, y, s = group_shift(severity)
        print(f"Severity: {severity:.1f}, Group A positive rate: {y[s==0].mean():.3f}, Group B positive rate: {y[s==1].mean():.3f}")
    
    print("\nCovariate Shift (Group A):")
    for severity in gammas:
        X, y, s = covariate_shift(severity=severity, group='A')
        print(f"Severity: {severity:.1f}, Group A feature 1 std: {X[s==0][:, 0].std():.3f}, Group B feature 1 std: {X[s==1][:, 0].std():.3f}")
    
    print("\nLabel Shift:")
    for severity in betas:
        X, y, s = label_shift(severity=severity)
        print(f"Severity: {severity:.2f}, Group A positive rate: {y[s==0].mean():.3f}, Group B positive rate: {y[s==1].mean():.3f}")