"""" Predict function for solvent"""
import pickle


def predict(df) -> int:
    """
    Predict the function of solvent.

    Parameters
    ----------
    df
        Dataframe on which we predict
    
    Return
    ------
    int

    """
    try:
        model = pickle.load(open("api_built.pkl","rb"))
        pred = model.predict(df)
        print(pred)
        return pred

    except FileNotFoundError:
        print("File not found")
