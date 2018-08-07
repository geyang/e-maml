.PHONY: build
build:
	docker-compose up --build
install-dc:
	sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
	sudo chmod +x /usr/local/bin/docker-compose
publish-docker:
	docker tag super-expert episodeyang/super-expert
	docker tag super-expert-gpu episodeyang/super-expert-gpu
	docker push episodeyang/super-expert
	docker push episodeyang/super-expert-gpu
