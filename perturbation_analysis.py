def perturbation_analysis(correct_sql, student_sql, model):
    base_latent_correct = model(correct_sql)
    base_latent_student = model(student_sql)
    base_distance = torch.norm(base_latent_correct - base_latent_student, dim=1)

    distance_changes = []
    for i in range(student_sql.size(1)):  # Iterate over the token positions
        perturbed_student = student_sql.clone()
        perturbed_student[:, i] = 366  # Replace token with UNKNOWN token

        perturbed_latent_student = model(perturbed_student)
        new_distance = torch.norm(base_latent_correct - perturbed_latent_student, dim=1)

        change_in_distance = (new_distance - base_distance).mean().item()
        distance_changes.append(change_in_distance)

    return distance_changes

# Analyze the effect of perturbing each token
changes = perturbation_analysis(correct_sql, student_sql, model)
print(f"Distance changes after perturbation: {changes}")