# Frontend Dockerfile
FROM node:14

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json
COPY ../package*.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Serve the application
RUN npm install -g serve
CMD ["serve", "-s", "build"]