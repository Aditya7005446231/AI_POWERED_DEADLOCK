def parse_input(data, is_matrix=False):
    if isinstance(data, str):
        data = data.strip()
        return eval(data) if is_matrix else list(map(int, data.strip('[]').split(',')))
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid input format.")



def is_safe_state(allocated, max_need, available):
    num_processes = len(allocated)
    num_resources = len(available)

    work = available[:]
    finish = [False] * num_processes
    safe_sequence = []

    while True:
        found = False
        for i in range(num_processes):
            if not finish[i]:
                need = [max_need[i][j] - allocated[i][j] for j in range(num_resources)]
                if all(need[j] <= work[j] for j in range(num_resources)):
                    for j in range(num_resources):
                        work[j] += allocated[i][j]
                    finish[i] = True
                    safe_sequence.append(f"P{i}")
                    found = True
        if not found:
            break

    if all(finish):
        return "Safe State", safe_sequence
    else:
        return "Deadlock Detected (Unsafe State)", []
