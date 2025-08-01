sudo apt-get update
sudo apt-get upgrade --yes
sudo apt-get install --yes zip unzip git build-essential libgl1 python3-venv python3-pip python-is-python3

mkdir --parents ~/.config/systemd/user
cp systemd/PurpleRanger@.service ~/.config/systemd/user
loginctl enable-linger $USER
systemd --user daemon-reload

echo "To use, enable the instance service you would like, using 'systemctl --user enable PurpleRanger@<pipeline_name>.service'"
echo
echo "For example 'systemctl --user enable PurpleRanger@vio.service'"
echo
echo "Finally run 'systemctl --user start PurpleRanger@<pipeline_name>.service'"