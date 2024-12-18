class MODEL():
    def __init__():
        return

validation_data = ...
metrics = ...
targets = ...


best_accuracy = 0
best_params = {"a" : 0, "b" : 0, "c" : 0}

for a in range(1,11):
    for b in range(1,11):
        for c in range(1,11):
            model = MODEL(a,b,c)
            preds = model.predict(validation_data)

            acc = metrics.accuracy(targets, preds)
            if(acc > best_accuracy):
                best_accuracy = acc
                best_params["a"] = a
                best_params["b"] = b
                best_params["c"] = c