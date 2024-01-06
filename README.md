# MNIST FOR DEPLOYMENT

This repo helps deploy MNIST on server environments.


## How to Run

`pip install -r requirements.txt`

`python3 app.py --port 8080`

## How to deploy on AWS

1. Create an Ubuntu based instance on AWS
2. clone this repo
3. do "How to Run"
4. port forward 8080 to 80 using `sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080`
5. Go to the "Security Group" associated with the instance and open tcp port 8080 in "Inbound Rules"
6. Try Access `http://<public_ip>:80`

## Future Works

1. add live camera feed and do inference every 1s (done)
2. add vae as well (to do)
3. remove torch from requirements.txt, because aws free tier crashes (done)
4. successfully deploy in aws (done) - camera not working in http; needs https
5. 


