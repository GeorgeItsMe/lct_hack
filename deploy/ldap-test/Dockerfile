FROM debian:bookworm-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends slapd ldap-utils && rm -rf /var/lib/apt/lists/*
COPY slapd.conf entries.ldif /test/
CMD ["sh", "-c", "mkdir -p /var/lib/contour-test && slapadd -f /test/slapd.conf -l /test/entries.ldif && exec slapd -f /test/slapd.conf -h ldaps:/// -d 256"]
