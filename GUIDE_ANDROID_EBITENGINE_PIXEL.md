# Adapter un projet Go + Ebitengine en application Android pour Pixel

Ce document est un guide de travail complet pour transformer une démo ou un jeu écrit en Go avec Ebitengine en application Android installable sur un Google Pixel. Il est basé sur l’adaptation réelle de `go-cuddlymenu`, compilée, installée et validée sur un Pixel 10a connecté en USB.

Le but est double :

- pouvoir reconstruire ce projet sans rechercher les commandes ou les chemins d’outils ;
- disposer d’une méthode réutilisable pour les prochaines adaptations Go + Ebitengine vers Android.

> État de référence : 4 septembre 2026. Les versions Android, Gradle et Ebitengine évoluent. Les versions indiquées ici sont celles qui ont été validées ensemble. Pour un nouveau projet, commencer par les reproduire avant de tenter une mise à jour.

## 1. La voie rapide pour ce dépôt

Sur la machine où cette adaptation a été réalisée, le Pixel étant branché et autorisé :

```sh
cd /Users/olivier/GolandProjects/go-cuddlymenu
./scripts/run-android.sh
```

Ce script effectue toute la chaîne :

1. détection du SDK Android et de Java 17 ;
2. génération d’une bibliothèque Android Ebitengine pour `arm64-v8a` ;
3. compilation de l’APK de débogage avec Gradle ;
4. vérification qu’un seul appareil Android est autorisé ;
5. installation ou mise à jour de l’application avec `adb install -r` ;
6. arrêt de l’ancienne instance ;
7. lancement de `com.olivierh.cuddlymenu/.MainActivity`.

Résultats générés :

```text
android/app/libs/cuddlymenu.aar
android/app/libs/cuddlymenu-sources.jar
android/app/build/outputs/apk/debug/app-debug.apk
```

Les fichiers de `android/app/libs/` et les répertoires de build sont ignorés par Git : ce sont des produits de compilation, pas des sources.

Pour lancer seulement la version ordinateur :

```sh
go run ./cmd/cuddlymenu
```

Pour valider le code Go sans produire d’APK :

```sh
go test ./...
go vet ./...
```

## 2. Comprendre la chaîne de construction

Ebitengine ne génère pas directement un APK complet. Le projet est assemblé en deux couches :

```text
Code du jeu Go
    │
    ├── version ordinateur
    │      cmd/cuddlymenu/main.go
    │          └── ebiten.RunGame(...)
    │
    └── pont mobile
           mobile/mobile.go
               └── mobile.SetGame(...)
                       │
                       ▼
              ebitenmobile bind
                       │
                       ▼
          cuddlymenu.aar + libgojni.so
                       │
                       ▼
        projet Android Java + Gradle/AGP
                       │
                       ▼
                 app-debug.apk
                       │
                       ▼
              adb install → Pixel
```

Les responsabilités sont les suivantes :

| Élément | Rôle |
|---|---|
| Go | Compile le jeu et gère ses modules. |
| Ebitengine | Boucle de jeu, rendu 2D, audio et entrées clavier/tactiles. |
| `ebitenmobile` | Génère le pont Android et appelle `gomobile`. |
| `gomobile` | Compile le code Go en bibliothèques natives Android. Il est utilisé indirectement. |
| NDK Android | Fournit le compilateur natif qui produit `libgojni.so`. |
| AAR | Archive Android contenant les classes Java Ebitengine et la bibliothèque native Go. |
| Java 17 | Exécute Gradle et compile l’activité Android. |
| Gradle Wrapper | Télécharge et exécute la version exacte de Gradle du projet. |
| Android Gradle Plugin, ou AGP | Transforme le projet Android et l’AAR en APK. |
| SDK Platform / Build Tools | Fournit l’API Android, `aapt`, `zipalign`, `apksigner`, etc. |
| Platform Tools | Fournit principalement `adb`, utilisé pour communiquer avec le téléphone. |

La documentation officielle d’Ebitengine confirme ce modèle : `ebitenmobile bind` produit un AAR contenant une vue Android `EbitenView`, qui doit ensuite être placée dans une application Android native : [Ebitengine — Mobile](https://ebitengine.org/en/documents/mobile.html).

## 3. Configuration exacte validée

### 3.1 Machine de développement

| Composant | Version validée | Emplacement sur cette machine |
|---|---:|---|
| macOS | Darwin 25.6, Apple Silicon | — |
| Go | 1.27.1 `darwin/arm64` | `/opt/homebrew/bin/go` |
| Dépôt | module `go-cuddlymenu` | `/Users/olivier/GolandProjects/go-cuddlymenu` |
| Homebrew | installation Apple Silicon | `/opt/homebrew/bin/brew` |
| Java/OpenJDK | 17.0.20.1 | `/opt/homebrew/opt/openjdk@17` |
| Binaire Java réel | 17 | `/opt/homebrew/opt/openjdk@17/bin/java` |
| SDK Android | installation Homebrew | `/opt/homebrew/share/android-commandlinetools` |
| `sdkmanager` | Command-line Tools | `/opt/homebrew/bin/sdkmanager` |
| `adb` | Platform Tools 37.0.1 | `/opt/homebrew/share/android-commandlinetools/platform-tools/adb` |
| SDK Platform | Android 36 | `/opt/homebrew/share/android-commandlinetools/platforms/android-36` |
| Build Tools | 35.0.0 et 36.0.0 | `/opt/homebrew/share/android-commandlinetools/build-tools/` |
| NDK | 28.2.13676358 | `/opt/homebrew/share/android-commandlinetools/ndk/28.2.13676358` |
| `ebitenmobile` global | version de développement, non utilisée par le script | `/Users/olivier/go/bin/ebitenmobile` |
| Cache des modules Go | dépendances téléchargées | `/Users/olivier/go/pkg/mod` |
| Cache Gradle | distributions et dépendances | `/Users/olivier/.gradle/` |

Attention à `/usr/bin/java` sur macOS : il peut n’être qu’un lanceur Apple affichant « Unable to locate a Java Runtime ». Pour ce projet, utiliser explicitement le JDK Homebrew ci-dessus ou définir correctement `JAVA_HOME`.

Commandes permettant de retrouver les chemins sur une autre machine :

```sh
command -v go
go version
go env GOPATH GOMODCACHE

command -v brew
brew --prefix
brew --prefix openjdk@17

command -v sdkmanager
command -v adb
command -v ebitenmobile

java -version
```

Le Gradle système n’est pas nécessaire. Le fichier `android/gradlew` est le point d’entrée à utiliser ; il garantit Gradle 8.11.1 et place sa distribution dans `~/.gradle/wrapper/dists/`.

`ebitenmobile` peut être installé globalement avec :

```sh
go install github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11
```

Il apparaît normalement dans :

```text
$(go env GOPATH)/bin/ebitenmobile
```

soit `/Users/olivier/go/bin/ebitenmobile` sur cette machine. Le script préfère tout de même `go run ...@v2.9.11`, car la version demandée est alors visible dans la commande et ne dépend pas de l’âge du binaire global.

### 3.2 Pixel réellement validé

| Propriété | Valeur observée |
|---|---|
| Constructeur | Google |
| Modèle | Pixel 10a |
| Nom interne | `stallion` |
| Numéro de série ADB | propre à l’appareil ; ne pas le versionner |
| Android | 17 |
| Niveau API du téléphone | 37 |
| ABI | `arm64-v8a` uniquement |
| Définition physique | `1080 × 2424` en portrait, donc `2424 × 1080` en paysage |
| Densité | 420 dpi |
| Taille de page mémoire observée | 4096 octets |

Commandes utilisées pour obtenir ces informations :

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb

"$ADB" devices -l
"$ADB" shell getprop ro.product.model
"$ADB" shell getprop ro.product.device
"$ADB" shell getprop ro.build.version.release
"$ADB" shell getprop ro.build.version.sdk
"$ADB" shell getprop ro.product.cpu.abi
"$ADB" shell getconf PAGE_SIZE
"$ADB" shell wm size
"$ADB" shell wm density
```

Le téléphone est en API 37 alors que l’application est compilée avec `compileSdk 36` et cible `targetSdk 36`. C’est normal pour ce test local : Android sait exécuter une application ciblant une API antérieure. Lors d’une publication future, vérifier les exigences actuelles du Play Store et la table de compatibilité API/AGP.

## 4. Installer les outils une seule fois sur macOS

### 4.1 Installation Homebrew utilisée ici

Sur un Mac Apple Silicon :

```sh
brew install go
brew install openjdk@17
brew install --cask android-commandlinetools
```

Le cask Android place ici le SDK sous :

```text
/opt/homebrew/share/android-commandlinetools
```

Sur un Mac Intel, le préfixe Homebrew est généralement `/usr/local` au lieu de `/opt/homebrew`. Ne recopier donc pas aveuglément les chemins : vérifier avec `brew --prefix`.

### 4.2 Variables d’environnement

Configuration temporaire pour le terminal courant :

```sh
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
```

Pour les rendre permanentes sous Zsh, placer ces lignes dans `~/.zprofile`, puis ouvrir un nouveau terminal ou exécuter :

```sh
source ~/.zprofile
```

`ANDROID_HOME` est la variable de référence recommandée. `ANDROID_SDK_ROOT` est aujourd’hui dépréciée par Android, même si le script l’accepte encore comme solution de compatibilité : [Android — variables d’environnement](https://developer.android.com/tools/variables).

Si Android Studio est installé à la place des outils Homebrew, les chemins les plus courants sur macOS sont :

```text
SDK Android :  ~/Library/Android/sdk
JDK intégré :  /Applications/Android Studio.app/Contents/jbr/Contents/Home
```

Il n’est pas nécessaire d’installer Android Studio pour cette procédure en ligne de commande.

### 4.3 Installer les composants du SDK

Afficher les composants disponibles :

```sh
sdkmanager --list
```

Installer l’ensemble utilisé par ce projet :

```sh
sdkmanager \
  "platform-tools" \
  "platforms;android-36" \
  "build-tools;36.0.0" \
  "ndk;28.2.13676358"
```

Accepter les licences :

```sh
sdkmanager --licenses
```

La syntaxe et les identifiants de paquets viennent de la documentation officielle : [Android — sdkmanager](https://developer.android.com/tools/sdkmanager).

Vérification :

```sh
test -x "$ANDROID_HOME/platform-tools/adb"
test -f "$ANDROID_HOME/platforms/android-36/android.jar"
test -x "$ANDROID_HOME/build-tools/36.0.0/aapt"
test -d "$ANDROID_HOME/ndk/28.2.13676358"
```

### 4.4 Ce qu’il ne faut pas installer manuellement

Les bibliothèques Go déclarées dans `go.mod` sont téléchargées automatiquement par Go. Il ne faut pas copier leur code dans le projet ni éditer le cache de modules.

Les dépendances directes de ce dépôt sont :

| Module | Version | Utilité |
|---|---:|---|
| `github.com/hajimehoshi/ebiten/v2` | 2.9.11 | moteur du jeu et support mobile |
| `github.com/olivierh59500/ym-player` | pseudo-version du 7 juin 2025 | lecture de la musique YM |

Les dépendances indirectes importantes sont :

| Module | Version | Utilité générale |
|---|---:|---|
| `github.com/ebitengine/gomobile` | pseudo-version du 23 septembre 2025 | génération et compilation du pont mobile |
| `github.com/ebitengine/oto/v3` | 3.4.1 | sortie audio multiplateforme |
| `github.com/ebitengine/purego` | 0.9.0 | appels natifs utilisés par Ebitengine/Oto |
| `github.com/ebitengine/hideconsole` | 1.0.0 | comportement de fenêtre sur certaines plateformes |
| `golang.org/x/sync` | 0.21.0 | primitives de synchronisation |
| `golang.org/x/sys` | 0.44.0 | interfaces système |

Les modules plus spécifiques au bureau, comme `github.com/jezek/xgb`, peuvent apparaître dans `go.mod` même si l’APK Android ne les embarque pas pour son ABI.

Commandes de gestion :

```sh
go mod download
go mod tidy
go list -m all
go mod graph
```

`go.sum` doit être versionné : il verrouille les sommes de contrôle des dépendances.

Le premier build nécessite un accès Internet pour remplir les caches Go et Gradle. Les builds suivants réutilisent généralement :

```text
/Users/olivier/go/pkg/mod
/Users/olivier/Library/Caches/go-build
/Users/olivier/.gradle/caches
/Users/olivier/.gradle/wrapper/dists
```

Il n’est pas nécessaire d’exécuter manuellement `gomobile init` dans cette procédure : `ebitenmobile bind` prépare et appelle la version de `gomobile` qui lui correspond.

## 5. Préparer le Pixel pour le développement USB

### 5.1 Activer les options développeur

Sur le Pixel :

1. ouvrir **Paramètres** ;
2. ouvrir **À propos du téléphone** ;
3. toucher sept fois **Numéro de build** ;
4. saisir le code de verrouillage si Android le demande ;
5. revenir dans **Système > Options pour les développeurs** ;
6. activer **Débogage USB**.

Brancher un câble USB capable de transporter des données. Certains câbles ne servent qu’à la charge.

Au premier branchement :

1. déverrouiller le téléphone ;
2. accepter la boîte de dialogue contenant l’empreinte RSA de l’ordinateur ;
3. cocher l’autorisation permanente si ce Mac est une machine de confiance.

Référence : [Android — exécuter une application sur un appareil physique](https://developer.android.com/studio/run/device).

### 5.2 Vérifier la connexion

Avec le chemin complet, qui fonctionne même si `adb` n’est pas dans `PATH` :

```sh
/opt/homebrew/share/android-commandlinetools/platform-tools/adb devices -l
```

Résultat attendu :

```text
List of devices attached
<SERIAL_PIXEL>  device  usb:... product:stallion model:Pixel_10a device:stallion
```

Interprétation de la deuxième colonne :

| État | Signification | Action |
|---|---|---|
| `device` | appareil prêt | continuer |
| `unauthorized` | empreinte RSA non acceptée | déverrouiller le Pixel et accepter la demande |
| `offline` | connexion ADB bloquée | débrancher/rebrancher puis redémarrer ADB |
| aucune ligne | appareil non détecté | vérifier câble, port USB, mode USB et débogage |

Redémarrer le serveur ADB si nécessaire :

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
"$ADB" kill-server
"$ADB" start-server
"$ADB" devices -l
```

Sur macOS, aucun pilote USB constructeur supplémentaire n’est normalement nécessaire.

## 6. Restructurer correctement le projet Go

### 6.1 Pourquoi un unique `package main` ne suffit pas

Sur ordinateur, le point d’entrée classique appelle :

```go
ebiten.RunGame(game)
```

Sur Android, le code du jeu doit être importable par un petit paquet mobile qui appelle :

```go
mobile.SetGame(game)
```

Un paquet `main` n’est pas conçu pour être importé. La structure réutilisable est donc :

```text
go-cuddlymenu/
├── assets/
│   ├── embed.go
│   └── menu/...
├── cmd/
│   └── cuddlymenu/
│       └── main.go             # exécutable ordinateur
├── menu/
│   ├── main.go                 # type Game, Update, Draw, Layout
│   ├── controls.go             # commandes tactiles
│   └── ...                     # logique du jeu
├── mobile/
│   └── mobile.go               # pont ebitenmobile
├── android/
│   └── ...                     # coque native Android
└── scripts/
    └── run-android.sh
```

Ici, tous les anciens fichiers de `menu/` ont été passés de :

```go
package main
```

à :

```go
package menu
```

Le point d’entrée ordinateur a été déplacé dans `cmd/cuddlymenu/main.go` :

```go
package main

import (
    "log"

    "github.com/hajimehoshi/ebiten/v2"
    "go-cuddlymenu/menu"
)

func main() {
    ebiten.SetWindowSize(768, 536)
    ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
    ebiten.SetWindowTitle("Cuddly Demos - Menu")
    if err := ebiten.RunGame(menu.NewGame()); err != nil {
        log.Fatal(err)
    }
}
```

Cette séparation conserve une seule implémentation du jeu pour toutes les plateformes.

### 6.2 Le paquet mobile minimal

Le fichier `mobile/mobile.go` contient :

```go
package mobile

import (
    enginemobile "github.com/hajimehoshi/ebiten/v2/mobile"
    "go-cuddlymenu/menu"
)

func init() {
    enginemobile.SetGame(menu.NewGame())
}

func Dummy() {}
```

Points importants :

- ne pas appeler `ebiten.RunGame` dans ce paquet ;
- `mobile.SetGame` remet l’objet `ebiten.Game` à la vue Android ;
- `Dummy` doit être exportée, car `gomobile` ignore un paquet qui n’expose aucun symbole ;
- le nom Go du paquet, ici `mobile`, entre dans le nom de la classe Java générée.

Avec :

```text
-javapkg com.olivierh.cuddlymenu
package Go : mobile
```

la vue générée est :

```java
com.olivierh.cuddlymenu.mobile.EbitenView
```

Cette correspondance est importante dans l’import de `MainActivity.java`.

## 7. Embarquer les ressources dans le binaire

### 7.1 Le problème des chemins relatifs

Sur ordinateur, ceci peut fonctionner :

```go
os.ReadFile("assets/menu/tiles.png")
```

Dans un APK Android, il n’existe pas de répertoire courant contenant l’arborescence du dépôt. Les images et la musique seraient absentes et le jeu utiliserait des placeholders ou échouerait.

### 7.2 La solution avec `go:embed`

Le fichier `assets/embed.go` embarque les fichiers à la compilation :

```go
package assets

import "embed"

//go:embed menu/*.png menu/*.ym
var Files embed.FS
```

Le chargeur du jeu lit ensuite cette ressource :

```go
data, err := gameassets.Files.ReadFile("menu/tiles.png")
```

Le paquet d’embed se trouve dans `assets/`, car un motif `go:embed` ne peut pas remonter avec `..` en dehors du répertoire de son paquet.

Règles pratiques pour les adaptations futures :

- embarquer les petites ressources indispensables au démarrage ;
- utiliser des chemins avec `/` via le paquet `path`, pas `filepath`, pour un `embed.FS` ;
- inclure explicitement chaque extension utile dans les motifs ;
- conserver une erreur ou un placeholder clair lorsqu’une ressource manque ;
- vérifier la taille finale de l’APK si les ressources sont volumineuses.

## 8. Verrouiller des versions compatibles

### 8.1 Versions de ce projet

Extrait de `go.mod` :

```go
module go-cuddlymenu

go 1.25.0

require (
    github.com/hajimehoshi/ebiten/v2 v2.9.11
    github.com/olivierh59500/ym-player v0.0.0-20250607015657-bb5818debd02
)
```

Mise à jour utilisée :

```sh
go get github.com/hajimehoshi/ebiten/v2@v2.9.11
go mod tidy
go test ./...
```

### 8.2 Toujours aligner Ebitengine et `ebitenmobile`

Utiliser la même version pour la bibliothèque et l’outil :

```sh
go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11 ...
```

Éviter pour un build reproductible :

```sh
ebitenmobile ...
```

si l’on ne sait pas avec quelle version le binaire global a été compilé.

Éviter aussi :

```sh
go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@latest ...
```

dans un script stable : `latest` peut changer sans modification du dépôt.

Un décalage de versions a produit pendant cette adaptation des erreurs Java telles que :

```text
cannot find symbol: class Renderer
cannot find symbol: usesStrictContextRestoration()
cannot find symbol: onContextLost()
cannot find symbol: setRenderer(...)
```

Cause : l’outil global générait une vue Java plus récente que le paquet Ebitengine 2.7.4 lié dans le jeu.

Une seconde tentative avec l’ancien outil 2.7.4 et Go 1.27 a échoué dans `golang.org/x/tools` :

```text
invalid array length -delta * delta
```

Cause : les outils d’analyse Go de cette ancienne version n’étaient plus compatibles avec Go 1.27. Le passage coordonné à Ebitengine 2.9.11 a réglé les deux problèmes.

Principe réutilisable : mettre à jour ensemble le module Ebitengine et la version `@...` de la commande, puis relancer tous les tests.

## 9. Ajouter des commandes tactiles fiables

### 9.1 Séparer l’intention de commande de la source physique

La physique du personnage ne doit pas savoir si une commande provient du clavier, d’un écran tactile ou d’une souris. Elle reçoit seulement des booléens :

```go
type controlState struct {
    Left  bool
    Right bool
    Fly   bool
}
```

Les entrées sont fusionnées :

```go
left := ebiten.IsKeyPressed(ebiten.KeyLeft) || controls.Left
right := ebiten.IsKeyPressed(ebiten.KeyRight) || controls.Right
thrust := ebiten.IsKeyPressed(ebiten.KeyUp) || controls.Fly
```

Cette approche garde la version ordinateur fonctionnelle et facilite l’ajout ultérieur d’une manette.

### 9.2 Lire tous les doigts à chaque tick

Pour une commande maintenue, il faut interroger l’état courant de tous les contacts :

```go
g.touchIDs = ebiten.AppendTouchIDs(g.touchIDs[:0])
for _, id := range g.touchIDs {
    x, y := ebiten.TouchPosition(id)
    state.press(layout, x, y)
}
```

Pourquoi :

- `AppendTouchIDs` renvoie chaque doigt encore posé ;
- `TouchPosition(id)` donne sa position logique ;
- une lecture par frame permet de maintenir gauche/droite ou le vol ;
- le parcours de tous les identifiants fournit naturellement le multitouch ;
- réutiliser `g.touchIDs[:0]` évite une allocation de slice à chaque tick.

Il ne faut pas limiter ce cas à un événement « just pressed » : le personnage ne bougerait que pendant une frame.

La référence des API tactiles se trouve dans [la documentation du paquet Ebitengine](https://pkg.go.dev/github.com/hajimehoshi/ebiten/v2#AppendTouchIDs).

### 9.3 Le multitouch

Chaque doigt met à jour le même état :

```go
func (s *controlState) press(layout controlLayout, x, y int) {
    if layout.Left.contains(x, y) {
        s.Left = true
    }
    if layout.Right.contains(x, y) {
        s.Right = true
    }
    if layout.Fly.contains(x, y) {
        s.Fly = true
    }
}
```

Ainsi un doigt peut rester sur droite pendant qu’un second maintient **VOLER**.

L’état est recréé vide à chaque tick. Dès qu’un doigt est levé, la commande correspondante disparaît ; il n’y a pas de bouton « collé ».

### 9.4 Zones de contact et dessin

Les boutons sont modélisés par un centre et un rayon :

```go
type controlButton struct {
    X      float64
    Y      float64
    Radius float64
}

func (b controlButton) contains(x, y int) bool {
    dx := float64(x) - b.X
    dy := float64(y) - b.Y
    return dx*dx+dy*dy <= b.Radius*b.Radius
}
```

Les valeurs validées ici sont :

```text
rayon gauche/droite : 42 unités logiques
rayon VOLER         : 50 unités logiques
```

Les cercles et les flèches sont dessinés avec :

```go
github.com/hajimehoshi/ebiten/v2/vector
```

Le bouton pressé change de couleur. Ce retour visuel aide énormément au diagnostic des touches et évite à l’utilisateur de se demander si son doigt est reconnu.

### 9.5 Prévisualisation à la souris

Le même système accepte le clic gauche :

```go
if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
    x, y := ebiten.CursorPosition()
    state.press(layout, x, y)
}
```

Cela permet de tester les hitboxes sur ordinateur sans reconstruire un APK. Sur bureau, les commandes sont affichées après le premier contact ou lorsque la fenêtre est plus large que la scène.

### 9.6 Piège visuel rencontré

Les zones tactiles gauche/droite étaient correctes, mais les deux chevrons avaient été dessinés en sens inverse. Les tests de logique ne pouvaient pas détecter ce problème sémantique.

Pour chaque portage, vérifier visuellement :

- bouton situé à gauche → icône `‹` ;
- bouton situé à droite → icône `›` ;
- appui gauche → personnage vers la gauche ;
- appui droit → personnage vers la droite ;
- direction + action simultanées ;
- couleur d’appui sur le bon bouton.

Une capture réelle du téléphone fait partie du test, pas uniquement `go test`.

## 10. Adapter l’affichage au format très large du Pixel

### 10.1 Le problème de rapport d’aspect

La démo originale possède une surface logique fixe de :

```text
768 × 536, rapport ≈ 1,43:1
```

Le Pixel 10a en paysage fournit :

```text
2424 × 1080, rapport ≈ 2,24:1
```

Étirer directement la démo déformerait les pixels et les sprites. Utiliser toute la largeur pour la scène cacherait aussi une partie du contenu ou obligerait à modifier la caméra.

La solution choisie :

- conserver le canvas original `768 × 536` ;
- calculer une largeur logique adaptée au ratio externe ;
- centrer le canvas original ;
- utiliser les deux bandes latérales pour le pad et le bouton **VOLER**.

### 10.2 Calcul de la largeur logique

```go
func logicalWidth(outsideWidth, outsideHeight int) int {
    if outsideWidth <= 0 || outsideHeight <= 0 {
        return 768
    }
    w := (outsideWidth*536 + outsideHeight - 1) / outsideHeight
    if w < 768 {
        return 768
    }
    if w > 1280 {
        return 1280
    }
    return w
}
```

Pour le Pixel :

```text
ceil(2424 × 536 / 1080) = 1204 unités logiques
```

La scène de 768 unités laisse donc environ :

```text
(1204 - 768) / 2 = 218 unités logiques de chaque côté
```

ce qui suffit pour les commandes tactiles.

### 10.3 `Layout` et coordonnées tactiles

```go
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
    g.layoutWidth = logicalWidth(outsideWidth, outsideHeight)
    return g.layoutWidth, 536
}
```

`Layout` retourne la taille logique du jeu. Ebitengine convertit ensuite les coordonnées tactiles physiques dans ce même repère logique. Les hitboxes et le dessin des boutons peuvent donc employer exactement les mêmes coordonnées.

La taille est plafonnée à 1280 pour éviter qu’une surface ultra-large crée des zones latérales démesurées.

### 10.4 Centrer la scène sans lui appliquer l’effet CRT

Le jeu est d’abord dessiné dans un canvas fixe, puis placé au centre :

```go
offsetX := (screen.Bounds().Dx() - screenWidth) / 2
op.GeoM.Translate(float64(offsetX), 0)
screen.DrawImage(source, &op)
```

Les commandes sont dessinées ensuite sur l’écran final. Elles ne reçoivent donc pas le shader CRT destiné à la démo.

### 10.5 Portrait, paysage, encoche et barres système

Le manifeste force une orientation paysage :

```xml
android:screenOrientation="sensorLandscape"
```

`sensorLandscape` autorise les deux sens horizontaux au lieu d’imposer uniquement un paysage fixe.

L’activité autorise l’utilisation des bords autour de l’encoche :

```java
attributes.layoutInDisplayCutoutMode =
        WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
```

Les boutons conservent une marge interne. Pour un autre téléphone, vérifier les deux rotations : l’encoche peut passer de gauche à droite.

## 11. Générer l’AAR Ebitengine

Commande exacte de ce projet :

```sh
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"

mkdir -p android/app/libs

go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11 \
  bind \
  -target android/arm64 \
  -androidapi 23 \
  -javapkg com.olivierh.cuddlymenu \
  -o android/app/libs/cuddlymenu.aar \
  ./mobile
```

Signification des options :

| Option | Effet |
|---|---|
| `@v2.9.11` | aligne l’outil sur la version Ebitengine de `go.mod` |
| `bind` | génère une bibliothèque utilisable depuis Android |
| `-target android/arm64` | ne compile que l’ABI du Pixel |
| `-androidapi 23` | API Android minimale utilisée lors de la compilation native |
| `-javapkg com.olivierh.cuddlymenu` | préfixe des classes Java générées |
| `-o .../cuddlymenu.aar` | destination de l’archive Android |
| `./mobile` | paquet Go contenant `mobile.SetGame` |

### Pourquoi cibler seulement `arm64`

Le Pixel 10a exposé par ADB ne propose que `arm64-v8a`. La première génération sans restriction d’ABI produisait un AAR d’environ 36 Mo avec :

```text
armeabi-v7a
arm64-v8a
x86
x86_64
```

La génération `android/arm64` produit ici un AAR d’environ 8,7 Mo et un APK de débogage d’environ 15 Mo.

Pour une diffusion générale ou des émulateurs, retirer la restriction ou ajouter les ABI voulues. Un APK `arm64-v8a` ne fonctionnera pas sur un émulateur `x86_64`.

### Contenu de l’AAR

Inspection :

```sh
unzip -l android/app/libs/cuddlymenu.aar
```

Éléments principaux :

```text
classes.jar
jni/arm64-v8a/libgojni.so
AndroidManifest.xml
```

`libgojni.so` contient le runtime Go, Ebitengine, le jeu et ses ressources embarquées.

## 12. La coque Android

### 12.1 Arborescence utile

```text
android/
├── build.gradle
├── settings.gradle
├── gradle.properties
├── gradlew
├── gradlew.bat
├── gradle/wrapper/
│   ├── gradle-wrapper.jar
│   └── gradle-wrapper.properties
└── app/
    ├── build.gradle
    ├── libs/
    │   └── cuddlymenu.aar       # généré, non versionné
    └── src/main/
        ├── AndroidManifest.xml
        ├── java/com/olivierh/cuddlymenu/MainActivity.java
        └── res/values/styles.xml
```

### 12.2 Réutiliser la coque pour un nouveau jeu

Le moyen le plus rapide consiste à copier le répertoire `android/` de ce dépôt dans le nouveau projet, sans copier les éléments générés :

```text
à conserver                    à régénérer
----------------------------  ------------------------------------------
android/gradlew                android/app/libs/<jeu>.aar
android/gradlew.bat            android/app/libs/<jeu>-sources.jar
android/gradle/wrapper/        android/app/build/
android/settings.gradle        android/.gradle/
android/build.gradle
android/gradle.properties
android/app/build.gradle
android/app/src/
```

Ensuite, rechercher toutes les occurrences spécifiques au projet :

```sh
rg -n 'cuddlymenu|Cuddly Demos|com\.olivierh' android scripts mobile
```

Valeurs à renommer ensemble :

| Emplacement | Ancienne valeur | Nouvelle valeur attendue |
|---|---|---|
| `go.mod` | `module go-cuddlymenu` | module du nouveau jeu |
| imports Go | `go-cuddlymenu/menu` | nouveau chemin de module/package |
| option `-javapkg` | `com.olivierh.cuddlymenu` | préfixe Java choisi |
| `namespace` Gradle | `com.olivierh.cuddlymenu` | même namespace Android |
| `applicationId` Gradle | `com.olivierh.cuddlymenu` | identifiant unique de l’application |
| dossier Java | `java/com/olivierh/cuddlymenu/` | chemin correspondant au package Java |
| déclaration Java | `package com.olivierh.cuddlymenu;` | même package |
| import d’`EbitenView` | `...cuddlymenu.mobile.EbitenView` | préfixe Java + nom du paquet Go mobile |
| script ADB | package et activité actuels | nouveaux identifiants |
| label manifeste | `Cuddly Demos` | nom visible de l’application |

Le `namespace` et l’`applicationId` peuvent techniquement différer, mais les garder identiques simplifie les builds internes. L’`applicationId` doit être unique pour éviter qu’une adaptation remplace une autre application déjà installée sur le Pixel.

Si aucun wrapper Gradle ne peut être copié, on peut en générer un une fois avec une installation Gradle compatible :

```sh
gradle -p android wrapper \
  --gradle-version 8.11.1 \
  --distribution-type bin
```

Puis versionner les quatre éléments générés :

```text
android/gradlew
android/gradlew.bat
android/gradle/wrapper/gradle-wrapper.jar
android/gradle/wrapper/gradle-wrapper.properties
```

Après cela, l’installation Gradle globale n’est plus nécessaire.

### 12.3 Dépôts de plugins et de dépendances

`android/settings.gradle` indique où Gradle peut obtenir AGP et les bibliothèques Android :

```groovy
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "CuddlyMenu"
include(":app")
```

Le dépôt `google()` est indispensable pour Android Gradle Plugin. `mavenCentral()` couvre les dépendances Java générales. Le fichier AAR du jeu reste une dépendance locale et n’est envoyé sur aucun dépôt.

### 12.4 Versions Gradle et AGP

`android/build.gradle` :

```groovy
plugins {
    id "com.android.application" version "8.10.1" apply false
}
```

`android/gradle/wrapper/gradle-wrapper.properties` fixe :

```text
distributionUrl=https\://services.gradle.org/distributions/gradle-8.11.1-bin.zip
```

Cette combinaison utilise :

```text
Android Gradle Plugin 8.10.1
Gradle 8.11.1
JDK 17
compileSdk 36
```

AGP 8.10 prend officiellement en charge l’API 36 et demande au minimum Gradle 8.11.1 et JDK 17 : [Android — notes d’AGP 8.10](https://developer.android.com/build/releases/agp-8-10-0-release-notes).

Toujours employer :

```sh
./android/gradlew -p android ...
```

et non un `gradle` global, qui pourrait avoir une version incompatible.

### 12.5 Configuration de l’application

Parties importantes de `android/app/build.gradle` :

```groovy
android {
    namespace "com.olivierh.cuddlymenu"
    compileSdk 36

    defaultConfig {
        applicationId "com.olivierh.cuddlymenu"
        minSdk 23
        targetSdk 36
        versionCode 1
        versionName "1.0"

        ndk {
            abiFilters "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
}

dependencies {
    implementation files("libs/cuddlymenu.aar")
}
```

Différence entre les niveaux SDK :

| Paramètre | Rôle |
|---|---|
| `minSdk 23` | version Android minimale installable |
| `targetSdk 36` | comportement Android pour lequel l’application déclare être conçue |
| `compileSdk 36` | API disponible au compilateur Java/AGP |

La valeur `-androidapi 23` d’`ebitenmobile` et `minSdk 23` de Gradle sont alignées volontairement.

### 12.6 Manifeste

Le manifeste :

- déclare OpenGL ES 2.0 ;
- rend `MainActivity` lançable ;
- force le paysage ;
- annonce les changements de configuration gérés par l’activité ;
- applique le thème plein écran.

Extraits :

```xml
<uses-feature
    android:glEsVersion="0x00020000"
    android:required="true" />

<activity
    android:name=".MainActivity"
    android:configChanges="keyboard|keyboardHidden|orientation|screenLayout|screenSize|smallestScreenSize|uiMode"
    android:exported="true"
    android:screenOrientation="sensorLandscape">
```

## 13. Initialisation et cycle de vie Android

### 13.1 Ordre correct dans `MainActivity`

Avec Ebitengine 2.9, l’ordre fiable est :

1. `Seq.setContext(getApplicationContext())` ;
2. configurer la fenêtre et le cutout ;
3. construire `EbitenView` ;
4. lui donner le focus ;
5. appeler `setContentView(ebitenView)` ;
6. seulement ensuite masquer les barres système.

Extrait :

```java
Seq.setContext(getApplicationContext());

ebitenView = new EbitenView(this);
ebitenView.setFocusableInTouchMode(true);
ebitenView.requestFocus();
setContentView(ebitenView);
hideSystemUi();
```

Ebitengine 2.9 et antérieur demandent explicitement `Seq.setContext`. La documentation indique qu’Ebitengine 2.10 le fera automatiquement ; conserver l’appel reste sans danger : [Ebitengine — Mobile](https://ebitengine.org/en/documents/mobile.html).

### 13.2 Ne pas masquer les barres trop tôt

Pendant cette adaptation, appeler :

```java
getWindow().getInsetsController()
```

avant `setContentView` a provoqué :

```text
java.lang.NullPointerException
PhoneWindow.getInsetsController(...)
MainActivity.hideSystemUi(...)
MainActivity.onCreate(...)
```

La `DecorView` n’existait pas encore. La version corrigée obtient le contrôleur après l’installation du contenu :

```java
WindowInsetsController controller =
        getWindow().getDecorView().getWindowInsetsController();
```

et teste toujours `controller != null`.

### 13.3 Suspendre et reprendre Ebitengine

```java
@Override
protected void onPause() {
    if (ebitenView != null) {
        ebitenView.suspendGame();
    }
    super.onPause();
}

@Override
protected void onResume() {
    super.onResume();
    hideSystemUi();
    if (ebitenView != null) {
        ebitenView.resumeGame();
    }
}
```

Sans ces appels, l’audio ou la boucle de rendu peuvent continuer en arrière-plan ou reprendre incorrectement.

Le mode immersif est réappliqué dans `onWindowFocusChanged`, car Android peut faire réapparaître les barres système après une boîte de dialogue ou un changement de focus.

### 13.4 Ne pas ouvrir l’audio trop tôt

Le paquet mobile est initialisé au chargement de `libgojni.so`, avant que `MainActivity` ait fini d’installer le contexte Android et la vue Ebitengine.

La première version appelait `initAudio()` depuis `NewGame()`. Sur le Pixel, le chargement est resté bloqué pendant l’ouverture OpenSLES/Oboe et l’activité est restée blanche.

La correction consiste à créer le jeu sans ouvrir le périphérique audio, puis à initialiser l’audio au premier `Update` :

```go
func (g *Game) Update() error {
    if !g.audioReady {
        g.audioReady = true
        g.initAudio()
    }

    // suite de la logique...
}
```

Règle générale : dans la fonction appelée par `mobile.SetGame`, éviter toute initialisation dépendant d’un service Android pas encore disponible. Différer au premier `Update` :

- ouverture audio ;
- vibration ;
- presse-papiers ;
- appels nécessitant une fenêtre ou un contexte Android ;
- chargement réseau bloquant.

Les calculs Go purs, le décodage de petites ressources embarquées et la construction de l’état de jeu restent adaptés à `NewGame`.

## 14. Construire l’APK

Une fois l’AAR présent :

```sh
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"

./android/gradlew -p android --console=plain assembleDebug
```

APK produit :

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

Un APK `debug` est automatiquement signé avec la clé de débogage Android et peut être installé directement. Référence : [Android — construire depuis la ligne de commande](https://developer.android.com/build/building-cmdline).

Nettoyage si Gradle conserve un état incohérent :

```sh
./android/gradlew -p android clean
./android/gradlew -p android --console=plain assembleDebug
```

Ne pas supprimer `android/gradle/wrapper/` : il fait partie des sources reproductibles du projet.

## 15. Installer et lancer manuellement

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
APK=android/app/build/outputs/apk/debug/app-debug.apk

"$ADB" install -r "$APK"
"$ADB" shell am force-stop com.olivierh.cuddlymenu
"$ADB" shell am start -n com.olivierh.cuddlymenu/.MainActivity
```

Options utiles :

| Commande | Rôle |
|---|---|
| `adb install -r` | remplace l’application en conservant ses données |
| `adb install -r -d` | autorise aussi une baisse de `versionCode`, si nécessaire |
| `adb uninstall com.olivierh.cuddlymenu` | désinstalle et efface les données |
| `adb shell am force-stop ...` | garantit le redémarrage du nouveau code natif |
| `adb shell am start -n ...` | lance explicitement l’activité |

Si plusieurs appareils sont connectés, sélectionner le numéro de série :

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
PIXEL_SERIAL="remplacer-par-le-numero-affiche-par-adb-devices"

"$ADB" -s "$PIXEL_SERIAL" install -r "$APK"
"$ADB" -s "$PIXEL_SERIAL" shell am start \
  -n com.olivierh.cuddlymenu/.MainActivity
```

Le script fourni exige volontairement exactement un appareil en état `device`, afin de ne jamais installer par erreur sur un autre téléphone.

## 16. Vérifications après compilation

### 16.1 Vérifier le manifeste, les SDK et l’ABI

```sh
AAPT=/opt/homebrew/share/android-commandlinetools/build-tools/36.0.0/aapt
APK=android/app/build/outputs/apk/debug/app-debug.apk

"$AAPT" dump badging "$APK"
```

Éléments observés et attendus :

```text
package: name='com.olivierh.cuddlymenu'
sdkVersion:'23'
targetSdkVersion:'36'
compileSdkVersion='36'
launchable-activity: name='com.olivierh.cuddlymenu.MainActivity'
native-code: 'arm64-v8a'
uses-gl-es: '0x20000'
```

### 16.2 Vérifier la signature de l’APK

```sh
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export PATH="$JAVA_HOME/bin:$PATH"

APKSIGNER=/opt/homebrew/share/android-commandlinetools/build-tools/36.0.0/apksigner
"$APKSIGNER" verify --verbose \
  android/app/build/outputs/apk/debug/app-debug.apk
```

### 16.3 Vérifier l’alignement 16 Kio

Même si le Pixel testé utilise actuellement des pages de 4096 octets, une application contenant du code natif doit préparer la compatibilité avec les appareils à pages de 16 Kio.

Le projet utilise :

```text
AGP 8.10.1
NDK r28.2
```

Le NDK r28 génère par défaut des segments ELF compatibles 16 Kio et AGP moderne sait empaqueter correctement les bibliothèques natives. Référence : [Android — prise en charge des pages de 16 Kio](https://developer.android.com/guide/practices/page-sizes).

Vérification effectuée :

```sh
ZIPALIGN=/opt/homebrew/share/android-commandlinetools/build-tools/36.0.0/zipalign

"$ZIPALIGN" -c -P 16 -v 4 \
  android/app/build/outputs/apk/debug/app-debug.apk
```

Résultat attendu pour la bibliothèque Go :

```text
lib/arm64-v8a/libgojni.so (OK)
Verification successful
```

### 16.4 Vérifier que l’application est réellement au premier plan

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb

"$ADB" shell pidof com.olivierh.cuddlymenu
"$ADB" shell dumpsys activity activities | \
  grep -E 'topResumedActivity|com.olivierh.cuddlymenu'
```

Un PID non vide et `topResumedActivity=...MainActivity` confirment que le processus est actif et visible.

### 16.5 Capturer l’écran réel du Pixel

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb

"$ADB" exec-out screencap -p > /tmp/cuddlymenu-pixel.png
open /tmp/cuddlymenu-pixel.png
```

Vérifier sur la capture :

- démo centrée et non étirée ;
- aucune barre système persistante ;
- pad à gauche, `‹` puis `›` ;
- bouton **VOLER** à droite ;
- absence de découpe par l’encoche ;
- texte et sprites nets ;
- retour visuel d’un bouton pressé.

## 17. Tester le tactile

### 17.1 Test manuel recommandé

Sur le téléphone :

1. maintenir `›` ;
2. vérifier que le personnage part à droite ;
3. sans lever ce doigt, maintenir **VOLER** avec l’autre main ;
4. vérifier le déplacement diagonal et l’animation de poussée ;
5. relâcher uniquement **VOLER** et vérifier la chute ;
6. tester le même scénario avec `‹` ;
7. retourner le téléphone de 180 degrés en paysage et recommencer.

### 17.2 Test automatisé de la géométrie

`menu/controls_test.go` vérifie notamment :

- le calcul de largeur en portrait et en paysage ;
- le plafond pour les écrans ultra-larges ;
- le placement des commandes dans les bandes latérales ;
- la combinaison direction + vol simulant le multitouch.

```sh
go test ./menu -run 'Test(LogicalWidth|WideControlLayout|ControlState)' -v
```

Ces tests ne remplacent pas le contrôle visuel du sens des icônes.

## 18. Diagnostiquer rapidement avec ADB

### 18.1 Effacer les anciens logs et reproduire

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
PACKAGE=com.olivierh.cuddlymenu
ACTIVITY=com.olivierh.cuddlymenu/.MainActivity

"$ADB" logcat -c
"$ADB" shell am force-stop "$PACKAGE"
"$ADB" shell am start -n "$ACTIVITY"
sleep 3
"$ADB" logcat -d -v threadtime | tail -300
```

### 18.2 Filtrer sur le processus courant

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
PID=$("$ADB" shell pidof com.olivierh.cuddlymenu | tr -d '\r')

"$ADB" logcat -d --pid="$PID" -v threadtime
```

Filtre orienté crash :

```sh
"$ADB" logcat -d -v threadtime | \
  grep -Ei 'FATAL EXCEPTION|AndroidRuntime|panic|GoLog|cuddlymenu'
```

### 18.3 Observer les logs en direct

```sh
"$ADB" logcat -v color
```

Interrompre avec `Ctrl+C`.

## 19. Catalogue des problèmes rencontrés

### Écran blanc, aucune vue du jeu

Causes constatées :

1. ouverture audio dans `NewGame`, donc pendant le chargement de la bibliothèque native ;
2. crash de `MainActivity` avant `setContentView` ;
3. ancienne instance non redémarrée après remplacement de l’APK.

Actions :

```sh
adb logcat -c
adb shell am force-stop com.olivierh.cuddlymenu
adb shell am start -n com.olivierh.cuddlymenu/.MainActivity
adb logcat -d | grep -Ei 'FATAL|AndroidRuntime|GoLog'
```

Puis :

- différer les services Android/audio au premier `Update` ;
- vérifier l’ordre `setContentView` puis mode immersif ;
- vérifier que `Seq.setContext` est appelé avec Ebitengine 2.9 ;
- réinstaller et forcer l’arrêt avant le lancement.

### `cannot find symbol Renderer` pendant `ebitenmobile bind`

Cause : versions différentes entre le module Ebitengine du jeu et le générateur `ebitenmobile`.

Solution :

```sh
grep 'hajimehoshi/ebiten' go.mod
go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11 ...
```

La version après `@` doit correspondre à `go.mod`.

### `invalid array length -delta * delta`

Cause : ancien `golang.org/x/tools` utilisé avec une version de Go trop récente.

Solution préférée : mettre Ebitengine et `ebitenmobile` à jour ensemble, plutôt que bricoler un remplacement indirect de `x/tools`.

### `SDK location not found`

Solution :

```sh
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
```

ou créer localement `android/local.properties` :

```properties
sdk.dir=/opt/homebrew/share/android-commandlinetools
```

`local.properties` contient une configuration propre à la machine et ne doit pas être versionné.

### `Unable to locate a Java Runtime`

Le `/usr/bin/java` de macOS n’est pas un JDK utilisable.

```sh
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export PATH="$JAVA_HOME/bin:$PATH"
"$JAVA_HOME/bin/java" -version
```

### `adb: command not found`

Utiliser le chemin complet :

```sh
/opt/homebrew/share/android-commandlinetools/platform-tools/adb devices -l
```

ou ajouter `platform-tools` au `PATH`.

### `unauthorized`

- déverrouiller le Pixel ;
- accepter l’empreinte RSA ;
- si aucune fenêtre n’apparaît, révoquer les autorisations de débogage USB dans les options développeur, puis reconnecter.

### `ADB server didn't ACK` ou `Operation not permitted`

Cette erreur peut apparaître dans un terminal isolé ou un bac à sable qui interdit l’ouverture du port local ADB ou l’accès USB.

Actions :

- exécuter la commande dans le Terminal macOS normal ;
- autoriser explicitement l’accès USB/réseau local dans l’outil d’automatisation ;
- vérifier qu’un autre processus ADB ne tourne pas avec une version différente ;
- relancer `adb kill-server` puis `adb start-server` hors du bac à sable.

### `Unable to strip ... libgojni.so`

Gradle peut afficher :

```text
Unable to strip the following libraries, packaging them as they are: libgojni.so
```

Pour le build `debug` validé, l’APK reste fonctionnel. Contrôler néanmoins pour une release :

- la taille de l’APK/AAB ;
- l’alignement 16 Kio ;
- la présence des symboles souhaités ;
- les réglages de packaging et la version du NDK.

### Avertissement sur la version XML du SDK

Un avertissement disant qu’un outil comprend seulement une ancienne version du XML SDK indique généralement un décalage entre Command-line Tools et AGP.

```sh
sdkmanager --update
./android/gradlew -p android --version
```

Mettre à jour de façon coordonnée et relancer un build propre. Ne pas ignorer cet avertissement pour une release, même si le debug fonctionne.

### Ressources absentes dans l’APK

Vérifier :

```sh
grep -R 'go:embed' assets
go test ./...
unzip -l android/app/libs/cuddlymenu.aar | head
```

Ne pas revenir à des chemins relatifs dépendant du répertoire courant.

### Le clic marche mais pas le multitouch

Vérifier que le code :

- parcourt tous les `TouchID` ;
- ne conserve pas un identifiant unique global ;
- reconstruit l’état à chaque `Update` ;
- ne remplace pas `left/right` par l’état du dernier doigt ;
- emploie `OR` pour fusionner les contacts ;
- n’utilise pas seulement `JustPressedTouchIDs`.

### Le bouton est visible mais ne réagit pas au bon endroit

Causes probables :

- dessin en pixels physiques, hitbox en coordonnées logiques ;
- calcul de layout différent entre `Update` et `Draw` ;
- origine de scène non prise en compte ;
- rotation ou inset ayant changé la surface externe.

Ici, dessin et hitboxes passent tous deux par `makeControlLayout`, et les touches sont lues dans le repère logique Ebitengine.

## 20. Le script d’automatisation fourni

`scripts/run-android.sh` centralise les choix reproductibles :

```text
Ebitengine       2.9.11
ABI              arm64
API native min   23
package Java     com.olivierh.cuddlymenu
AAR              android/app/libs/cuddlymenu.aar
APK              android/app/build/outputs/apk/debug/app-debug.apk
activité         com.olivierh.cuddlymenu/.MainActivity
```

Détection automatique actuelle :

1. variables `ANDROID_HOME` ou `ANDROID_SDK_ROOT` si déjà définies ;
2. SDK Homebrew `/opt/homebrew/share/android-commandlinetools` ;
3. `JAVA_HOME` si défini ;
4. JDK d’Android Studio s’il existe ;
5. OpenJDK 17 Homebrew.

Le script s’arrête explicitement si :

- le SDK ou `adb` manque ;
- Java 17 manque ;
- le wrapper Gradle manque ;
- zéro ou plusieurs appareils sont autorisés ;
- une étape de build ou d’installation échoue.

Le `set -eu` empêche le script de continuer après une erreur ou avec une variable non définie.

## 21. Méthode réutilisable pour le prochain projet

### Phase A — audit du projet

- [ ] Identifier la version de Go : `go version`.
- [ ] Identifier la version Ebitengine dans `go.mod`.
- [ ] Trouver le type qui implémente `Update`, `Draw` et `Layout`.
- [ ] Relever toutes les entrées clavier/souris/manette.
- [ ] Relever toutes les lectures de fichiers par chemin relatif.
- [ ] Repérer l’audio ou les services ouverts dans le constructeur.
- [ ] Mesurer la taille logique et le rapport d’aspect du jeu.
- [ ] Exécuter `go test ./...` avant modification.

### Phase B — rendre le jeu importable

- [ ] Déplacer le jeu hors de `package main`.
- [ ] Créer `cmd/<nom>/main.go` pour le bureau.
- [ ] Exposer `NewGame()` ou un objet implémentant `ebiten.Game`.
- [ ] Vérifier à nouveau la version ordinateur.

### Phase C — ressources

- [ ] Créer un paquet `assets` utilisant `go:embed`.
- [ ] Remplacer les `os.ReadFile` relatifs par `embed.FS.ReadFile`.
- [ ] Inclure images, sons, polices, cartes et shaders externes.
- [ ] Vérifier que l’application fonctionne depuis un autre répertoire courant.

### Phase D — mobile

- [ ] Créer `mobile/mobile.go`.
- [ ] Appeler `mobile.SetGame(...)`.
- [ ] Ajouter une fonction exportée `Dummy`.
- [ ] Différer l’audio et les services dépendant du contexte.
- [ ] Épingler exactement la version d’`ebitenmobile`.

### Phase E — tactile et écran

- [ ] Créer un état de commande indépendant du périphérique.
- [ ] Lire tous les `TouchID` à chaque tick.
- [ ] Supporter au moins deux doigts simultanés.
- [ ] Partager la géométrie entre dessin et hit-test.
- [ ] Prévoir un retour visuel pressé/non pressé.
- [ ] Ajouter un fallback souris pour le bureau.
- [ ] Calculer un layout logique adapté au ratio du téléphone.
- [ ] Exploiter les bandes latérales plutôt que déformer la scène.
- [ ] Tester encoche, barres système et deux rotations paysage.

### Phase F — Android

- [ ] Créer le projet Gradle et son wrapper.
- [ ] Choisir ensemble JDK, Gradle, AGP et `compileSdk` compatibles.
- [ ] Aligner `minSdk` et `-androidapi`.
- [ ] Déclarer l’ABI voulue.
- [ ] Importer l’AAR local.
- [ ] Créer `MainActivity` avec `Seq.setContext` si nécessaire.
- [ ] Relier `onPause`/`onResume` à `suspendGame`/`resumeGame`.
- [ ] Installer la vue avant d’activer le mode immersif.
- [ ] Déclarer l’orientation et les changements de configuration.

### Phase G — validation réelle

- [ ] `go test ./...`.
- [ ] `go vet ./...`.
- [ ] Générer l’AAR pour la bonne ABI.
- [ ] `assembleDebug`.
- [ ] Contrôler `aapt dump badging`.
- [ ] Contrôler `zipalign -P 16`.
- [ ] Installer avec `adb install -r`.
- [ ] Forcer l’arrêt et relancer.
- [ ] Contrôler le PID et l’activité au premier plan.
- [ ] Vérifier les logs sans crash.
- [ ] Prendre une capture réelle.
- [ ] Tester le multitouch sur le téléphone.
- [ ] Refaire un lancement après mise en veille/reprise.

## 22. Passer d’un APK de développement à une diffusion

La procédure de ce guide produit un APK `debug`. Pour publier ou distribuer proprement :

- choisir un `applicationId` définitif et unique ;
- ajouter une icône adaptative et les ressources de marque ;
- incrémenter `versionCode` à chaque livraison ;
- mettre à jour `versionName` ;
- ne pas utiliser la clé de débogage ;
- créer et sauvegarder hors du dépôt un keystore de release ;
- configurer `signingConfigs` via des secrets locaux ou CI ;
- produire de préférence un Android App Bundle avec `bundleRelease` pour le Play Store ;
- réintroduire les ABI nécessaires à la diffusion ;
- vérifier les exigences `targetSdk` du moment ;
- tester les pages 16 Kio et plusieurs tailles d’écran ;
- exécuter `lint`, les tests et l’analyse APK/AAB ;
- vérifier licences, politique de confidentialité et droits sur les ressources.

Exemples de tâches Gradle :

```sh
./android/gradlew -p android lintDebug
./android/gradlew -p android assembleRelease
./android/gradlew -p android bundleRelease
```

Ne jamais versionner le mot de passe ou le keystore de production dans Git.

## 23. Références officielles

- [Ebitengine — Mobile et `ebitenmobile bind`](https://ebitengine.org/en/documents/mobile.html)
- [Ebitengine v2.9.11 — API Go](https://pkg.go.dev/github.com/hajimehoshi/ebiten/v2@v2.9.11)
- [Ebitengine — `AppendTouchIDs`](https://pkg.go.dev/github.com/hajimehoshi/ebiten/v2#AppendTouchIDs)
- [Android — exécuter sur un appareil physique](https://developer.android.com/studio/run/device)
- [Android — construire une application en ligne de commande](https://developer.android.com/build/building-cmdline)
- [Android — `sdkmanager`](https://developer.android.com/tools/sdkmanager)
- [Android — variables d’environnement](https://developer.android.com/tools/variables)
- [Android — notes de version AGP 8.10](https://developer.android.com/build/releases/agp-8-10-0-release-notes)
- [Android — compatibilité des versions du plugin Gradle](https://developer.android.com/build/releases/about-agp)
- [Android — configurer le NDK](https://developer.android.com/studio/projects/configure-agp-ndk)
- [Android — compatibilité avec les pages mémoire de 16 Kio](https://developer.android.com/guide/practices/page-sizes)
- [Go — installer la toolchain](https://go.dev/doc/install)

## 24. Résumé opérationnel

Pour gagner du temps lors de la prochaine adaptation, retenir surtout ceci :

1. rendre le jeu importable et conserver un `cmd/...` séparé pour le bureau ;
2. embarquer les ressources avec `go:embed` ;
3. créer un paquet mobile minuscule avec `mobile.SetGame` et `Dummy` ;
4. épingler exactement la même version d’Ebitengine et d’`ebitenmobile` ;
5. différer l’audio et les services Android jusqu’au premier `Update` ;
6. lire tous les doigts à chaque tick et partager la géométrie dessin/hitbox ;
7. conserver le canvas original et utiliser les marges du téléphone pour les commandes ;
8. utiliser le wrapper Gradle, jamais une version système inconnue ;
9. automatiser AAR → APK → ADB dans un script qui échoue clairement ;
10. ne jamais conclure sur le seul message « installation réussie » : contrôler logs, PID, capture et multitouch sur l’appareil réel.
