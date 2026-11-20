# iPhone prin USB în Camo Studio pentru FastGears

Ghidul folosește un iPhone și un laptop cu Windows. Pregătește un cablu USB de date compatibil cu ambele dispozitive. Pentru iPhone 17, folosește USB‑C la capătul telefonului și USB‑C sau USB‑A la laptop, în funcție de port. [Ghid Apple pentru conectarea prin cablu](https://support.apple.com/en-euro/guide/iphone/iph42d9b3178/ios)

## 1. Instalează aplicațiile

1. Pe iPhone, instalează **Camo Camera** de la Reincubate. Deschide aplicația și acordă-i acces la cameră când iOS îl solicită. [Camo Camera în App Store](https://apps.apple.com/ro/app/camo-camera/id1514199064)
2. Pe laptop, instalează **Camo Studio** și deschide-l din meniul Start. [Camo Studio în Microsoft Store](https://apps.microsoft.com/detail/9PGM3QB3PDRD)
3. Parcurge configurarea inițială. Dacă versiunea instalată din Store solicită **Compatibility add-on**, instalează-l din Camo Studio. Acesta face camera virtuală disponibilă pentru aplicații ca FastGears și poate cere permisiuni de administrator. Dacă ai închis configurarea, opțiunea se găsește și în notificările Camo Studio. [Instalarea pe Windows în manualul Camo](https://camo.com/support/camo/camo-getting-started#installing-on-windows)

## 2. Conectează iPhone-ul prin cablu

1. Deblochează iPhone-ul și conectează-l direct la un port USB al laptopului cu cablul de date.
2. Ține **Camo Camera** deschisă pe iPhone și **Camo Studio** pornit pe Windows. Conexiunea USB se face automat. Codul QR din **Pair Camo Camera** este pentru Wi‑Fi. [Conectarea unui iPhone în manualul Camo](https://camo.com/support/camo/camo-getting-started#using-camo-with-an-iphone)
3. În Camo Studio, deschide fila **Video** din stânga, apoi meniul **Device** și selectează iPhone-ul. Verifică dacă imaginea se mișcă în previzualizarea din centru. În **Lens**, poți selecta **Selfie 1x** pentru camera frontală, dacă opțiunea este disponibilă. [Setările video Camo](https://camo.com/support/camo/adjusting-video-settings#video)

Dacă iOS afișează **Trust This Computer?**, Camo poate funcționa și fără aprobarea acestei solicitări. **Trust** permite laptopului să acceseze și date de pe telefon. [Manualul Camo](https://camo.com/support/camo/camo-getting-started#using-camo-with-an-iphone) · [Ce înseamnă Trust în iOS](https://support.apple.com/en-us/109054)

## 3. Alege fundalul natural, fără blur

1. În fila **Video** din stânga, mergi la secțiunea **Background**.
2. Alege **Basic**, prima opțiune din stânga în interfața Windows folosită la test. În versiunile care afișează meniul **Mode**, opțiunea echivalentă se numește **Normal**. Verifică în previzualizare că fundalul real se vede clar. [Modurile de fundal din manualul Camo](https://camo.com/support/camo/adjusting-video-settings#image-enhancements)
3. Pentru testele FastGears, setează **Filter** și **Effects** pe **None**, iar **Retouch** și **Spotlight** pe **Off**. Dacă schimbi scena în Camo Studio, verifică din nou aceste opțiuni, deoarece unele scene aplică efecte proprii. [Ajustarea imaginii în Camo](https://camo.com/support/camo/adjusting-video-settings)
4. Pentru configurația folosită la test, selectează profilul **720p / 30 fps** din bara de sus, dacă este disponibil. Încadrează complet fața în previzualizare. [Profiluri video în Camo](https://camo.com/support/camo/camo-getting-started#selecting-a-video-profile)

## 4. Verifică imaginea în FastGears

După [configurarea FastGears](README.md#setup), rulează din rădăcina repository-ului, în PowerShell:

```powershell
.\.venv\Scripts\python.exe -m fastgears --camera 1 --camera-backend msmf --no-objects --no-llm
```

Deschide [dashboard-ul FastGears](http://127.0.0.1:8000/) și verifică dacă imaginea este aceeași cu cea din Camo Studio. Pe laptopul de test, `1` era Camo și `0` era camera HP. Dacă apare altă cameră pe laptopul tău, oprește FastGears cu **Ctrl+C** și schimbă indexul `--camera`. În timpul testului Camo, ține oprită instanța FastGears pentru camera HP.

## Probleme frecvente

- **Telefonul nu apare în Device.** Încearcă alt port USB sau alt cablu cu transfer de date și redeschide Camo Camera pe telefon. [Depanarea conexiunii USB în Windows](https://support.apple.com/en-us/108643)
- **Camo Studio are imagine, dar FastGears nu.** Confirmă că add-on-ul de compatibilitate este instalat și că ai ales indexul Camo cu `--camera-backend msmf`. Dacă MSMF deschide camera, dar fluxul se întrerupe, oprește FastGears, rulează `$env:OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS = '0'` în aceeași fereastră PowerShell și pornește din nou comanda de mai sus. [Setarea Media Foundation în documentația OpenCV](https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html)
