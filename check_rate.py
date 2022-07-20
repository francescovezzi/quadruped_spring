land_rate = 0
step_rate = 0
substep_rate = 0
for step in range(100):
    step_rate += 1
    for substep in range(10):
        substep_rate += 1
        if substep%5 == 0:
            land_rate += 1
print(step_rate)
print(substep_rate)
print(land_rate)