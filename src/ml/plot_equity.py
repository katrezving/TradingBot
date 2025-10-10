import json
import matplotlib.pyplot as plt

def main(path=\"models/report_rf.json\"):
    with open(path, \"r\") as f:
        report = json.load(f)
    eq = report[\"test_back\"][\"equity_curve\"]
    plt.figure(figsize=(10,5))
    plt.plot(eq)
    plt.title(\"Evolución del Equity (Test)\")
    plt.xlabel(\"Paso\")
    plt.ylabel(\"Equity\")
    plt.grid(True)
    plt.show()

if __name__ == \"__main__\":
    main()
