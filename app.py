from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def plot_bode(frequencies, impedance_real, impedance_imag):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.loglog(frequencies, np.abs(impedance_real + 1j * impedance_imag), '-o')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (Ohm)')
    plt.title('Bode Plot - Magnitude')

    plt.subplot(2, 1, 2)
    plt.semilogx(frequencies, np.angle(impedance_real + 1j * impedance_imag) * (180 / np.pi), '-o')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.title('Bode Plot - Phase')

    plt.tight_layout()
    plt.show()

def fit_circuit(frequencies, impedance_real, impedance_imag):
    # Define the equivalent circuit model function
    def circuit_model(f, Rb, R_sei, CPE_sei, R_ct, CPE_dl, W):

        # Define the impedance components
        Z_sei = R_sei + 1 / (1j * 2 * np.pi * f)**CPE_sei
        Z_dl = 1 / (1j * 2 * np.pi * f)**CPE_dl
        Z_ct = R_ct / (1 + 1j * 2 * np.pi * f * R_ct * CPE_dl)
        Z_w = W * np.sqrt(1j * 2 * np.pi * f)

        # Calculate total impedance
        Z_total = Rb + Z_sei + Z_ct + Z_dl + Z_w

        return Z_total

    # Fit the circuit model to the impedance data
    p0 = (1, 1, 1, 1, 1, 1)  # Initial guess for parameters
    popt, pcov = curve_fit(circuit_model, frequencies, impedance_real + 1j * impedance_imag, p0=p0)

    return popt

def calculate_soh(Rb, Rb_max):
    return (Rb / Rb_max) * 100

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    data = pd.read_csv(file, header=None)  # Read CSV without headers

    frequencies = data.iloc[:, 0]  # Assume the first column is frequencies
    impedance_real = data.iloc[:, 1]  # Assume the second column is the real part of impedance
    impedance_imag = data.iloc[:, 2]  # Assume the third column is the imaginary part of impedance

    # Plot Bode plot
    plot_bode(frequencies, impedance_real, impedance_imag)

    # Fit equivalent circuit model
    circuit_parameters = fit_circuit(frequencies, impedance_real, impedance_imag)

    # Calculate State-of-Health (SoH)
    Rb = circuit_parameters[0]  # Assuming the first parameter is the battery resistance
    Rb_max = 100  # Maximum battery resistance (example value)
    soh = calculate_soh(Rb, Rb_max)

    return jsonify({'circuit_parameters': circuit_parameters, 'soh': soh})

if __name__ == '__main__':
    app.run(debug=True)

