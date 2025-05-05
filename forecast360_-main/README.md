# Forecast360

Forecast360 is a comprehensive web application designed to provide real-time weather forecasting, predictive analytics, and insightful visualization. Built with modern web technologies and backed by powerful AI/ML models, Forecast360 empowers users with accurate and user-friendly weather updates.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## About

Forecast360 offers a streamlined, interactive platform to:

- Access current weather conditions.
- Get forecasts based on geolocation.
- Visualize data trends over time.
- Predict future weather patterns using machine learning models.

The project was developed as part of an academic and professional portfolio to showcase expertise in full-stack development, API integration, and machine learning.

## Features

- Real-time weather data fetching.
- AI/ML based predictive weather forecasting.
- User authentication and profile management.
- Interactive data visualization (charts, graphs).
- Responsive UI for mobile and desktop.
- Secure handling of sensitive credentials.

## Tech Stack

- **Frontend:** Flutter (Mobile), React (Web)
- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL
- **Authentication:** JWT (JSON Web Tokens)
- **Cloud Services:** Google Cloud Platform (GCP)
- **Storage:** Firebase / AWS S3 (for media uploads)
- **DevOps:** GitHub Actions, Docker (for containerization)
- **Security:** Argon2 Password Hashing, Secure Secret Management

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js and npm
- Flutter SDK (for mobile app development)
- Docker (optional, for local containerization)
- Git

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/Nevedha4/forecast360_.git
cd forecast360_/backend
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file for backend configuration.

4. Run the FastAPI server:

```bash
uvicorn main:app --reload
```

### Frontend Setup (Web)

1. Navigate to the project directory:

```bash
cd forecast360_/frontend
```

2. Install npm dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

### Mobile App (Flutter)

1. Navigate to mobile directory:

```bash
cd forecast360_/mobile
```

2. Get Flutter dependencies:

```bash
flutter pub get
```

3. Run the application:

```bash
flutter run
```

## Project Structure

```plaintext
forecast360_
|-- backend/            # FastAPI backend services
|-- frontend/           # React frontend
|-- mobile/             # Flutter mobile app
|-- model/              # ML models and scripts
|-- assets/             # Images, icons, other static assets
|-- README.md
|-- LICENSE
```

## Contributing

We welcome contributions! Please fork the repository and submit a pull request.

Steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).


> Designed and Developed by team with precision and passion.

