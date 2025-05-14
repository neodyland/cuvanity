#!/bin/sh
if [ -z "$ADDR" ]; then
    echo "Error: ADDR environment variable is not set."
    exit 1
fi

HIDDEN_SERVICE_DIR="/work/hidden_service"
HIDDEN_SERVICE_REAL_DIR="/work/hidden_service_real"
cat <<EOF > /etc/tor/torrc
HiddenServiceDir $HIDDEN_SERVICE_REAL_DIR/
HiddenServicePort 80 $ADDR
EOF

MISSING=0
check_file() {
    FILE="$1"
    if [ ! -f "$HIDDEN_SERVICE_DIR/$FILE" ]; then
        echo "Missing file: $HIDDEN_SERVICE_DIR/$FILE"
    MISSING=1
    fi
}

if [ ! -d "$HIDDEN_SERVICE_DIR" ]; then
    echo "Directory $HIDDEN_SERVICE_DIR does not exist."
    exit 1
fi

FILES="hostname
hs_ed25519_public_key
hs_ed25519_secret_key"
for FILE in $FILES; do
    check_file "$FILE"
done

if [ "$MISSING" -ne 0 ]; then
    echo "Some required files are missing."
    exit 2
fi

if [ ! -d "$HIDDEN_SERVICE_REAL_DIR" ]; then
    echo "Copying hidden service files to $HIDDEN_SERVICE_REAL_DIR"
    mkdir -p "$HIDDEN_SERVICE_REAL_DIR"
    for FILE in $FILES; do
        cp "$HIDDEN_SERVICE_DIR/$FILE" "$HIDDEN_SERVICE_REAL_DIR/"
    done
    echo "Copying done."
    chmod 700 "$HIDDEN_SERVICE_REAL_DIR"
fi

tor -f /etc/tor/torrc