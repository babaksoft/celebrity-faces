
def make_prediction():
    return { "prediction": "Positive" }


def predict():
    return make_prediction()


def main():
    prediction = predict()
    print(prediction)


if __name__ == '__main__':
    main()
