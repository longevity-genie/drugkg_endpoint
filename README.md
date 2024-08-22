# biocypher_endpoint
Endpoint for KG-RAG AI assistant based on just-agents and fastAPI libraries

## Cloning the repo
Clone the project with dependencies:
```commandline
git clone --recurse-submodules https://github.com/winternewt/biocypher_endpoint.git
```
## Deploying the docker environment
Install latest docker and docker-compose from respective sites. If needed, you may use or refer to the provided script 
```commandline
install_docker_ubuntu.sh
```

Create environment variable file and populate the OPENAI_API_KEY with respective key
```commandline
cp env.local.template .env.local
nano .env.local
```

Build the chat-ui and bring up the docker-compose environment: 
```commandline
docker-compose up --build
```

## Setting up python backend and library

```commandline
micromamba create -f environment.yaml
micromamba activate biochatter_api
```

## Starting REST APIs
```
python index.py
```
