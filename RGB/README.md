# Analiza culorii faciale în FastGears

Modulul măsoară culoarea obrajilor, frunții și buzelor față de reperul personal. Dashboard-ul afișează măsurătorile CIELAB și observațiile despre culoare în panoul **Culoare facială · RGB**.

## Pornește testul pe Camo

1. Conectează iPhone-ul prin cablu, configurează fundalul **Basic** și verifică imaginea în Camo Studio. [Configurarea Camo și a fundalului](../README_CAMO.md)
2. Din rădăcina proiectului, pornește FastGears cu opțiunea `--rgb`. Pe laptopul de test, Camo folosește indexul `1` și backendul `msmf`. Rulează o singură instanță FastGears în timpul probei.

   ```powershell
   .\.venv\Scripts\python.exe -m fastgears --camera 1 --camera-backend msmf --rgb --rgb-overlay --no-objects --no-llm
   ```

3. Deschide [dashboard-ul](http://127.0.0.1:8000/) și așază fața frontal, complet în cadru, cu gura relaxată. Fiecare regiune are nevoie de aproximativ **5 secunde de cadre valide** pentru reper. Apasă **Recalibrate** la schimbarea persoanei, camerei, poziției sau luminii.

În copia locală `test rgb`, dacă folosești mediul Python din folderul vecin, înlocuiește executabilul din comandă cu `& '..\AI_FastGears\.venv\Scripts\python.exe'`. Dacă Media Foundation întrerupe fluxul cu camera deschisă, folosește setarea OpenCV din ghidul Camo.

### Estimarea luminii în utilizarea obișnuită

Implicit, modulul folosește fruntea ca reper de lumină. În timpul calibrării memorează culoarea fiecărei regiuni și a reperului din aceleași cadre. La evaluare, estimează variațiile comune ale expunerii și balansului de alb în RGB liniar, apoi le compensează pe obraji și buze. Dashboard-ul afișează abaterile cromatice compensate (`compared_delta`), iar valorile brute (`delta`) rămân disponibile în `/api/state`.

O frunte acoperită sau indisponibilă, o variație `|ΔL*| > 9` pe frunte ori o schimbare de peste `6` unități în planul `a*b*` suspendă observațiile până când condițiile redevin comparabile. Lasă fața complet în cadru la calibrare.

### Martor extern opțional pentru probe controlate

Fixează un cartonaș **gri neutru mat** în cadru, lângă față, aproximativ în același plan și luminat de aceeași sursă. Alege un dreptunghi cuprins în interiorul cartonașului, în afara feței. Coordonatele sunt fracții din lățimea și înălțimea imaginii pe care o afișează FastGears. Originea este în stânga sus, inclusiv când oglindirea este activă.

Repornește cu `--rgb-reference "x0,y0,x1,y1"` și verifică dreptunghiul desenat de `--rgb-overlay`. Spre exemplu, `--rgb-reference "0.05,0.08,0.16,0.20"` selectează zona dintre 5–16% pe orizontală și 8–20% pe verticală. Ajustează valorile la poziția cartonașului. Lasă `RGB_REFERENCE` gol și omite opțiunea `--rgb-reference` pentru modul automat bazat pe frunte.

## Ce măsoară

| Observație | Semnal de intrare și prag inițial |
|---|---|
| Creștere persistentă a roșeții obrajilor | `Δa* ≥ +2` în ambii obraji |
| Reducere persistentă a colorației obrajilor | `Δa* ≤ −2` sau `ΔC* ≤ −2` în fiecare obraz |
| Schimbare persistentă a nuanței buzelor | `Δb* ≤ −2,5` spre albastru sau `ΔC* ≤ −3` spre gri |

Pragurile sunt exprimate în unități Lab și cresc cu variația la repaus. Ele sunt cel puțin `3,5 × MAD`, cu un minim de `0,5` pentru MAD. Observația cere **1 secundă** de persistență. Revenirea necesită **0,5 secunde** sub 65% din prag. Pentru reducerea colorației obrajilor se acceptă `|ΔL*| ≤ 8` după corecția luminii, iar pentru buze `≤ 10`. Roșeața se poate detecta și când `|ΔL*|` compensat trece de 8, până la 16, dacă diferența dintre variațiile de luminozitate ale celor doi obraji este de cel mult 7 și ambii depășesc pragul de roșeață ajustat, de cel puțin `Δa* = +3`. Dacă există un martor extern, observațiile se suspendă la `|ΔL*| > 6` sau deplasare cromatică de peste `4` unități la martor, ori `|ΔL*| > 9` pe frunte. Reperul rămâne fix până la resetare. Duratele de calibrare și persistență se pot modifica prin `RGB_CALIBRATION_S` și `RGB_PERSISTENCE_S`.

Sub măsurători, **Evaluări RGB individuale** arată pentru fiecare observație procentul din prag, dacă schimbarea persistă, revine sau este neevaluabilă. Pragul adaptiv folosește variația măsurătorilor compensate din timpul calibrării. Când evaluarea devine activă, cardul descrie schimbarea vizuală găsită. Cardul de roșeață afișează separat valoarea `Δa*` folosită la evaluare pentru fiecare obraz. La buze, rezultatul precizează direcția predominantă, spre albastru sau spre o cromatică mai scăzută.

OpenCV livrează pixeli BGR `uint8`. Modulul îi tratează ca sRGB, îi normalizează la 0–1 și calculează Lab în virgulă mobilă, unde `a*` și `b*` au zero real. Masca buzelor scade conturul interior al gurii și îndepărtează marginile. La fiecare regiune se raportează mediana, dispersia, numărul probelor și fracția respinsă. Metadatele de expunere și balans de alb nu sunt disponibile prin integrarea Camo folosită aici. [Conversia CIELAB în OpenCV](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)

Lumina colorată și umbrele pot afecta obrajii sau buzele diferit de frunte. O schimbare reală care afectează și fruntea poate fi atenuată de corecție. Machiajul schimbat, barba, ochelarii și ocluziile parțiale pot schimba pixelii selectați. Testele validează măsurarea și deciziile software pe imagini controlate. O validare a semnificației clinice ar necesita date și evaluare independente.

Pe laptopul de test, analiza RGB cu măștile active a durat aproximativ **25 ms mediană** și **31 ms la percentila 95** pe cadru sintetic eșantionat la 1280 × 720, în timpul rulării suitei de teste. Modulul analizează cel mult zece cadre pe secundă. Costul capturii, MediaPipe și codării JPEG se adaugă separat.

`GET /api/state` expune câmpul `rgb` cu datele de calitate, abaterile și evenimentele. Modulul analizează cadrul local înainte de desenarea contururilor și transmite dashboard-ului doar măsurătorile și evenimentele.
