# SIMPLE NAIVE BAYES WITH 4 FEATURES (Fever, Cough, BodyAche, TravelHistory)
 
# Dataset from your file
data = [
    ("High","Yes","Yes","Yes","Yes"),
    ("High","Yes","No","No","Yes"),
    ("Normal","No","No","No","No"),
    ("Normal","Yes","No","No","No"),
    ("High","Yes","Yes","No","Yes"),
    ("Normal","No","Yes","No","No"),
    ("High","No","Yes","Yes","Yes"),
    ("Normal","Yes","Yes","No","No")
]
 
# Helper: count occurrences of feature=value AND target=class
def count(f_index, value, target_value):
    c = 0
    for row in data:
        if row[f_index] == value and row[4] == target_value:
            c += 1
    return c
 
def predict(fever, cough, bodyache, travel):
    total = len(data)
    total_yes = sum(1 for r in data if r[4] == "Yes")
    total_no  = total - total_yes
 
    # Priors
    p_yes = total_yes / total
    p_no  = total_no  / total
 
    # Likelihoods
    p_fever_yes = count(0, fever, "Yes") / total_yes
    p_fever_no  = count(0, fever, "No")  / total_no
 
    p_cough_yes = count(1, cough, "Yes") / total_yes
    p_cough_no  = count(1, cough, "No")  / total_no
 
    p_body_yes = count(2, bodyache, "Yes") / total_yes
    p_body_no  = count(2, bodyache, "No")  / total_no
 
    p_travel_yes = count(3, travel, "Yes") / total_yes
    p_travel_no  = count(3, travel, "No")  / total_no
 
    # Posterior
    score_yes = p_yes * p_fever_yes * p_cough_yes * p_body_yes * p_travel_yes
    score_no  = p_no  * p_fever_no  * p_cough_no  * p_body_no  * p_travel_no
 
    return score_yes, score_no
 
# ===== Predict for your example =====
fever = "High"
cough = "No"
bodyache = "Yes"
travel = "Yes"   # You can change this
 
yes_score, no_score = predict(fever, cough, bodyache, travel)
 
print("Score Yes =", yes_score)
print("Score No  =", no_score)
 
if yes_score > no_score:
    print("\nPrediction: Patient HAS Flue (YES)")
else:
    print("\nPrediction: Patient does NOT have Flue (NO)")
  