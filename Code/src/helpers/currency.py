"""
Example usage:

    > usd_conversion = get_usd_conversion("/path/to/LI-Large_Trans.csv")
    > print(usd_conversion)

    {
        "US Dollar": 1.0,
        "Euro": 0.8533787417099838,
        "Yuan": 6.697677681891531,
        "Yen": 105.3976841187823,
        "UK Pound": 0.7739872068230277,
        "Brazil Real": 5.646327447497649,
        "Australian Dollar": 1.4127728666786938,
        "Canadian Dollar": 1.319260431085624,
        "Ruble": 77.79226317392629,
        "Mexican Peso": 21.1287988422576,
        "Rupee": 73.44399970830806,
        "Swiss Franc": 0.9149993127687566,
        "Shekel": 3.3769999188170305,
        "Saudi Riyal": 3.751098012020342,
        "Bitcoin": 8.333333333333333e-05,
    }

Or, provide the following argument to get the static currency
conversion rates (to skip generating the data each run):

    > usd_conversion = get_usd_conversion(
        "/path/to/LI-Large_Trans.csv",
        get_base_amlworld_data=True,
    )

"""

def extract_usd_conversion(
    currency_conversion: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Given the result of `get_usd_conversion`, that is given a
    dictionary that maps each currency to its currency conversion
    rates, where each set of conversion rates may be incomplete,
    attempt to return a complete set of currency conversion rates
    for U.S. dollars
    """
    raise NotImplementedError(
        "This 'extract_usd_conversion' has not been implemented, because the "
        "data from the small transaction set provides a complete set of "
        " U.S. dollar conversion rates."
    )


def get_usd_conversion(dataset_dir: str, get_base_amlworld_data=False) -> dict[str, float]:
    """
    Given the name of a dataset, returns a currency conversion
    dictionary that will convert every value into U.S. dollars. This
    function currently only supports the AMLworld dataset and will fail
    to extract conversion rates from any dataset whose header does not
    match the AMLworld CSV datasets
    """
    data_header_mismatch = "Currency Conversion header mismatch"

    # To speed up use of this function, the following can be returned,
    # which is the result of running the method against the
    # `LI-Small_Trans.csv` data
    if get_base_amlworld_data:
        return {
            "US Dollar": 1.0,
            "Euro": 0.8533787417099838,
            "Yuan": 6.697677681891531,
            "Yen": 105.3976841187823,
            "UK Pound": 0.7739872068230277,
            "Brazil Real": 5.646327447497649,
            "Australian Dollar": 1.4127728666786938,
            "Canadian Dollar": 1.319260431085624,
            "Ruble": 77.79226317392629,
            "Mexican Peso": 21.1287988422576,
            "Rupee": 73.44399970830806,
            "Swiss Franc": 0.9149993127687566,
            "Shekel": 3.3769999188170305,
            "Saudi Riyal": 3.751098012020342,
            "Bitcoin": 8.333333333333333e-05,
        }

    # If we can't assume that every currency converts to every other currency
    # in the dataset, we need to create a partial conversion map for each
    # currency, and then aggregate the results into a single map for U.S.
    # dollar conversions
    currencies = set()
    currency_conversion = {}
    with open(dataset_dir, "r", encoding="utf-8") as file:
        header = True
        for line in file:
            columns = line.strip().split(",")

            # Checks that each data set is formatted with the expected data in
            # the same position
            if header:
                header = False
                if (
                    columns[5] != "Amount Received" or
                    columns[6] != "Receiving Currency" or
                    columns[7] != "Amount Paid" or
                    columns[8] != "Payment Currency"
                ):
                    raise ValueError(data_header_mismatch)
                continue

            sent_amount = columns[7]
            sent_currency = columns[8]
            received_amount = columns[5]
            received_currency = columns[6]

            currencies.add(sent_currency)
            currencies.add(received_currency)

            # To convert the sent currency to the received currency, multiply
            # it by this value
            conversion_rate = float(received_amount)/float(sent_amount)

            if sent_currency not in currency_conversion:
                currency_conversion[sent_currency] = {
                    sent_currency: 1.0
                }
            if received_currency not in currency_conversion[sent_currency]:
                currency_conversion[sent_currency][
                    received_currency
                ] = conversion_rate

    # See if we have a complete currency conversion map for U.S. dollars
    usd_conversion = currency_conversion["US Dollar"]
    if set(usd_conversion.keys()) == currencies:
        return usd_conversion
    return extract_usd_conversion(currency_conversion)
