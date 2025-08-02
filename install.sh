#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade --yes

# Install dependencies
sudo apt-get install --yes zip unzip git build-essential libgl1 python3-venv python3-pip python-is-python3

# Enable user service linger
sudo loginctl enable-linger $USER

# Install udev rule
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Copy service file
mkdir --parents ~/.config/systemd/user
cp systemd/PurpleRanger@.service ~/.config/systemd/user
systemctl --user daemon-reload

# Create Python venv
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Query user for team number
echo -n "Enter team number: "
read -r TEAM_NUMBER

sed -i -e "s/????/$TEAM_NUMBER/g" ~/.config/systemd/user/PurpleRanger@.service

echo
echo "To use, enable the instance service you would like, using 'systemctl --user enable PurpleRanger@<pipeline_name>.service'"
echo "For example 'systemctl --user enable PurpleRanger@vio.service'"
echo "Finally run 'systemctl --user start PurpleRanger@<pipeline_name>.service'"