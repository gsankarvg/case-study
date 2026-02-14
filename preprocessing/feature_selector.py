def select_features(df, dataset_name):

    dataset_name = dataset_name.lower()

    if dataset_name == "dayton":
        selected_columns = [
            "Electricity",
            "Temperature",
            "Relativehumidity",
            "Windspeed",
            "Month",
            "Day_of_week",
            "Hour"
        ]

    elif dataset_name == "houston":
        selected_columns = [
            "Electricity",
            "Temperature",
            "Month",
            "Day_of_week",
            "Hour"
        ]

    else:
        raise ValueError("Unknown dataset")

    return df[selected_columns]
