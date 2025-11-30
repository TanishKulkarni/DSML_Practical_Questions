True_Positive = 1
False_Positive = 1
False_Negative = 8
True_Negative = 90

Accuracy = (True_Positive + True_Negative) / (True_Positive + False_Positive + True_Negative + False_Negative)
Error_Rate = (False_Positive + False_Negative) / (True_Positive + False_Positive + True_Negative + False_Negative)
precision = True_Positive / (True_Positive + False_Positive)
Recall = True_Positive / (True_Positive + False_Negative)

print("Accuracy:", Accuracy * 100, "%")
print("Error Rate:", Error_Rate * 100, "%")
print("Precision:", precision * 100, "%")
print("Recall:", Recall * 100, "%")