from normalize import normalize_business_name, normalize_business_address
import numpy as np
import os

def run_tests():
    names = [
        "Orelee's Barbershop",
        "Custom Wealth Services LLC",
        "Consulting Nyasa Nursing Private Limited",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com",
        "Olszewski Holding Company LLC LLC",
        "CANDY'S-SERVICES",
        "heassociates.com",
        "LLC Moncada Léarning Center",
        "Ectolumdrex dba X+ Madison Inc",
        "அரிஹந்த் Foundation Private Limited",
        "New Delhi Techn0logies Private Limited"
    ]

    addresses = [
        "1795 Westchester Drive, High Point, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, Opp.Rta Office",
        "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ",
        None,
        np.nan,
        float('nan')
    ]

    os.makedirs('experiments', exist_ok=True)
    with open('experiments/test_normalization_output.txt', 'w', encoding='utf-8') as f:
        f.write("=== TESTING BUSINESS NAMES ===\n")
        for name in names:
            norm = normalize_business_name(name)
            f.write(f"RAW:        {name}\n")
            f.write(f"NORMALIZED: {norm}\n")
            f.write("-" * 50 + "\n")

        f.write("\n=== TESTING BUSINESS ADDRESSES ===\n")
        for addr in addresses:
            norm = normalize_business_address(addr)
            f.write(f"RAW:        {addr}\n")
            f.write(f"NORMALIZED: {norm}\n")
            f.write("-" * 50 + "\n")
        print("Tests completed successfully. Output saved to experiments/test_normalization_output.txt")

if __name__ == '__main__':
    run_tests()
