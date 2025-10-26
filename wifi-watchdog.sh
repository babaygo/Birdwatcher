#!/bin/bash

# Nom de l'interface Wi-Fi (souvent wlan0 ou wlan1)
INTERFACE="wlan0"

# Seuil en secondes avant reconnexion forcée
TIMEOUT=300  # 5 minutes

# Fichier temporaire pour stocker l'heure de dernière connexion
STATE_FILE="/tmp/wifi_last_seen"

# Vérifie si l'interface est connectée
is_connected() {
    nmcli -t -f DEVICE,STATE device | grep "$INTERFACE:connected" > /dev/null
}

# Met à jour le timestamp de dernière connexion
update_timestamp() {
    date +%s > "$STATE_FILE"
}

# Lit le timestamp précédent
last_seen() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo 0
    fi
}

# Boucle principale
while true; do
    if is_connected; then
        echo "Wi-Fi connecté"
        update_timestamp
    else
        echo "Wi-Fi non connecté"
        NOW=$(date +%s)
        LAST=$(last_seen)
        DELTA=$((NOW - LAST))

        if [ "$DELTA" -ge "$TIMEOUT" ]; then
            echo "Relance de l'interface Wi-Fi..."
            nmcli radio wifi off
            sleep 2
            nmcli radio wifi on
            update_timestamp
        else
            echo "Attente avant reconnexion forcée ($DELTA s)"
        fi
    fi

    sleep 60
done
