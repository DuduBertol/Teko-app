<img width="875" height="300" alt="image" src="https://github.com/user-attachments/assets/0a6c54c7-010c-41b2-8967-a09b29cf11fd" />


# Teko 💚🩸🏃‍♀️

> **body, rhythm, performance.**
>
> Teko combines menstrual phase and running data to generate a daily readiness score, a 7-day forecast, and Apple Health-powered insights.

![iOS](https://img.shields.io/badge/iOS-18.6%2B-black?style=flat-square&logo=apple)
![Swift](https://img.shields.io/badge/Swift-6-orange?style=flat-square&logo=swift)
![SwiftUI](https://img.shields.io/badge/SwiftUI-%40Observable-blue?style=flat-square)
![Core ML](https://img.shields.io/badge/Core%20ML-MLUpdateTask-purple?style=flat-square)
![HealthKit](https://img.shields.io/badge/HealthKit-Cycle%20%2B%20Running-red?style=flat-square)
![watchOS](https://img.shields.io/badge/watchOS-Companion-gray?style=flat-square)


<a href="https://apps.apple.com/br/app/teko-score-de-ciclo-e-corrida/id6761077930" target="_blank" rel="noopener noreferrer">
  <img 
      src="https://tools.applemediaservices.com/api/badges/download-on-the-app-store/black/pt-br?size=250x83&releaseDate=1276560000&h=7e7b68fad19738b5649a1bfb78ff46e9" 
      alt="Baixar na App Store" 
      style="width: 200px; height: auto;"
  >
</a>


---

<img width="9936" height="5376" alt="image" src="https://github.com/user-attachments/assets/4437d492-5d3c-4913-881b-08b71dddfad1" />

---

> **This repository is a technical showcase. Teko is closed-source and available on the App Store.**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Solution Overview](#solution-overview)
3. [Architecture](#architecture)
4. [Core ML Pipeline](#core-ml-pipeline)
5. [Feature Engineering](#feature-engineering)
6. [HealthKit Integration](#healthkit-integration)
7. [7-Day Forecast](#7-day-forecast)
8. [Apple Watch Companion](#apple-watch-companion)
9. [SwiftUI Architecture](#swiftui-architecture)
10. [Production Integrations](#production-integrations)
11. [Privacy Architecture](#privacy-architecture)
12. [Challenges & Lessons Learned](#challenges--lessons-learned)
13. [Tech Stack](#tech-stack)
14. [Image Assets Guide](#image-assets-guide)

---

## The Problem

Recovery and readiness apps like Whoop and Garmin Body Battery analyze sleep, heart rate variability, and activity load — but they treat every user identically. For female athletes, that's a significant gap.

**What's missing:**

- Menstrual cycle phase has a measurable, documented effect on running performance. Estrogen peaks near ovulation correlate with improved running economy and strength. The luteal phase raises resting heart rate and basal temperature, worsening endurance. Menstruation increases perceived exertion.
- No consumer app correlates cycle phase with running readiness in real time.
- Apps that do track cycle data (Clue, Flo) are siloed from workout data.
- Most health apps upload sensitive data to external servers.

**The opportunity:** Apple HealthKit already holds both menstrual cycle records and workout data. The intelligence layer just didn't exist yet.

---

## Solution Overview

Teko is a production iOS app that:

1. Reads menstrual cycle data and workout history from HealthKit (including third-party apps like Clue, Flo, Strava, and Runna).
2. Builds a 7-feature vector combining cycle physiology and running metrics.
3. Runs it through a Core ML neural network to produce a daily readiness score from 0 to 100.
4. Learns from the user's feedback over time via on-device fine-tuning — without ever sending data to a server.
5. Forecasts the next 7 days using predictable cycle progression.
6. Syncs the score to an Apple Watch companion app.

<img width="3840" height="1214" alt="image" src="https://github.com/user-attachments/assets/9b6e76ac-a2f6-49de-bea3-8368df336b09" />


---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Apple HealthKit                           │
│  MenstrualFlow · BasalBodyTemperature · HKWorkout · HeartRate    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Data Layer (Services)   │
              │  WorkoutStore               │
              │  CycleTrackingStore         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FeatureEngineeringService │
              │   7 features → RunCycle-    │
              │   Features (MLMultiArray)   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       ScoringService        │
              │  RunCycleScoreUpdatable     │
              │  (Core ML Neural Network)   │
              │  MLUpdateTask (on-device)   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼───────────────┐
              │    DailyScoreViewModel       │
              │   @Observable @MainActor     │
              │  scoreResult · forecast      │
              └──────┬───────────────┬───────┘
                     │               │
       ┌─────────────▼───┐    ┌───────▼─────────────────┐
       │   SwiftUI Views │    │ WatchConnectivityService│
       │  DailyScore-    │    │  applicationContext     │
       │  Screen         │    └───────┬─────────────────┘
       │  ForecastCard   │            │
       │  PhaseCard      │    ┌───────▼────────────────┐
       └─────────────────┘    │    watchOS App         │
                              │  WatchScoreView        │
                              │  WatchScoreDialRing    │
                              └────────────────────────┘
```

**Key architectural choices:**

| Decision | Rationale |
|---|---|
| `@Observable` + `@MainActor` | Modern Swift concurrency — no `ObservableObject` or `@Published` |
| Coordinator pattern | Single `enum Page` drives all navigation via `NavigationPath` |
| Services injected via `@Environment` | Testable, no singletons in view code |
| Parallel `async let` data fetching | HealthKit reads happen simultaneously, not sequentially |
| SwiftData for local persistence | `RunSession`, `TrainingExample`, `UserProfile` — all on-device |

---

## Core ML Pipeline

This is the technical heart of Teko.

### Why On-Device ML?

- **Privacy:** No health data ever leaves the device.
- **Offline:** Works without internet connectivity.
- **Personalization:** `MLUpdateTask` lets the model fine-tune to each individual user.
- **No server costs:** Inference and training are free.

### Model Selection: Why a Neural Network Regressor?

Core ML's `MLUpdateTask` (on-device fine-tuning) only supports two model types:
- Neural Networks (classification and regression)
- k-Nearest Neighbors (classification only)

The task is **regression** (predicting a continuous score 0–100), which eliminates k-NN. Gradient Boosted Trees would perform better on tabular data, but they cannot be updated on-device via CoreML.

| Model | Tabular Performance | CoreML Updatable | Regression Output |
|---|---|---|---|
| Gradient Boosted Tree | Excellent | No | Yes |
| k-Nearest Neighbors | Fair | Yes | No (classification only) |
| **Neural Network** | Good | **Yes** | **Yes** |

### Model Architecture

```
Input: MLMultiArray [7] (float32)
  │
Dense(7 → 32, ReLU)    ← normalization folded in; NOT updatable
  │
Dense(32 → 16, ReLU)   ← updatable
  │
Dense(16 → 1, Linear)  ← updatable (output = score 0–100)
```

### Normalization Folded Into Weights

A critical design decision: min-max normalization is **baked into the first dense layer's weights** at training time. This means the Swift app passes raw, unnormalized feature values — no preprocessing needed.

The math:
```
W1_fused = W1 × diag(1 / range)
b1_fused = W1 × (-min / range) + b1
```

**Why `dense_1` is frozen (not updatable):** If `dense_1` were updatable, the on-device SGD step would be scaled by raw input magnitudes (e.g., heart rate ≈ 150 bpm instead of ≈ 0.5 normalized). This causes gradient explosion. Only the downstream layers, which receive bounded post-ReLU activations, are safe to fine-tune.

### Synthetic Training Dataset

No public dataset exists that combines running metrics with menstrual cycle data — health data is extremely rare due to privacy regulations (GDPR/LGPD/HIPAA). The model ships with a **base trained on 500 synthetic samples** derived from peer-reviewed research.

The score formula has 8 layers:

| Layer | Effect | Source |
|---|---|---|
| Base score by phase | Follicular ~70, Ovulation ~86, Luteal ~58, Menstruation ~38 | McNulty et al. 2020 |
| Heavy flow penalty | Up to −10 points during menstruation | Hooper et al. 2011 |
| D4–D5 recovery bonus | +7 points as symptoms ease | Clinical observation |
| Follicular progression | Up to +12 points as estrogen rises | Beidleman et al. 1999 |
| Luteal deterioration | Up to −18 points through late luteal/PMS | Carmichael et al. 2021 |
| Temperature penalty | `max(0, delta) × 8 × 0.4` | Stephenson & Kolka 1993 |
| Fitness bonus | +0 / +2 / +4 for beginner/intermediate/advanced | Sims 2016 (ROAR) |
| Noise | Gaussian(0, 3) — simulates stress, sleep, diet | — |

**Dataset composition:**

| Phase | Samples | Mean Score | Range |
|---|---|---|---|
| Menstruation | 79 (15.8%) | 36 | 5–69 |
| Follicular | 114 (22.8%) | 77 | 54–98 |
| Ovulation | 47 (9.4%) | 87 | 76–98 |
| Luteal | 260 (52%) | 50 | 21–77 |

Luteal dominates at 52% because it lasts ~13 days per cycle. Ovulation is rare (~2–3 days). This reflects biological reality.

### Model Performance

| Metric | Value | Target |
|---|---|---|
| MAE | ~7.5 | < 8 |
| RMSE | ~9.2 | — |
| R² | ~0.77 | > 0.75 |
| Train/test MAE gap | < 5 | (no overfitting) |

### On-Device Fine-Tuning (MLUpdateTask)

After each run, the user rates how they felt. The app saves this as a `TrainingExample` and, once 5+ examples are collected, triggers a batch retrain:

```swift
// DailyScoreViewModel.swift
func retrainFromExamples(in modelContext: ModelContext) async {
    let examples = try modelContext.fetch(descriptor)
    guard examples.count >= minimumExamplesForTraining else { return }
    try await scoringService.retrainModel(from: examples)
}
```

| On-Device Training Config | Value |
|---|---|
| Updatable layers | `dense_2`, `dense_output` |
| Loss function | MSE |
| Optimizer | SGD (lr=0.001, batch=8) |
| Epochs per update | 5 |
| Minimum examples before first retrain | 5 |

The model does not retrain from scratch — it fine-tunes the pretrained weights. The base knowledge from the synthetic dataset is preserved; the user's real data gradually shifts the weights toward their personal physiology.

### Training Pipeline (Python → CoreML)

```
generate_dataset.py → training_data.csv → train_model.py → RunCycleScoreUpdatable.mlmodel
```

Built with PyTorch, converted to CoreML via `coremltools.NeuralNetworkBuilder`, then bundled into the Xcode project. The `.mlmodel` is compiled at build time by Xcode into a `.mlmodelc`.

---

## Feature Engineering

All feature construction lives in a single, pure, side-effect-free service:

```swift
// FeatureEngineeringService.swift
struct FeatureEngineeringService: Sendable {
    @MainActor
    static func buildFeatures(
        workoutStore: WorkoutStore,
        cycleStore: CycleTrackingStore
    ) -> RunCycleFeatures {
        let cycleDay = cycleStore.currentCycleDay()
        let phase = CyclePhase.estimate(cycleDay: cycleDay, cycleLength: cycleStore.averageCycleLength)
        return RunCycleFeatures(
            cycleDay: Float(cycleDay),
            phase: Float(phase.rawValue),
            flowIntensity: Float(cycleStore.todayFlowIntensity),
            basalTempDelta: Float(cycleStore.basalTempDelta),
            avgDistance: Float(workoutStore.averageDistance),
            avgPace: Float(workoutStore.averagePace),
            avgHeartRate: Float(workoutStore.averageHeartRate)
        )
    }
}
```

**The 7 features:**

| # | Feature | HealthKit Source | Notes |
|---|---|---|---|
| 0 | `cycle_day` | `HKCategoryType(.menstrualFlow)` + cycle start metadata | Days since last period start + 1 |
| 1 | `phase` | Derived from cycle day | Proportional boundaries per cycle length |
| 2 | `flow_intensity` | `HKCategoryValueMenstrualFlow` | Remapped: HK raw 1–4 → model 0–3 |
| 3 | `basal_temp_delta` | `HKQuantityType(.basalBodyTemperature)` | `today_temp - mean(last_30_readings)`. Fallback: 0.0 |
| 4 | `avg_distance` | `HKWorkout.totalDistance` | Mean of last 4 running workouts (km) |
| 5 | `avg_pace` | Derived from `duration / distance` | Mean pace across last 4 runs (min/km) |
| 6 | `avg_heart_rate` | `HKQuantityType(.heartRate)` during workout | Mean HR across last 4 runs. Fallback: 150 bpm |

**Phase estimation** uses proportional boundaries relative to cycle length (not fixed days), so a 24-day cycle and a 35-day cycle are handled correctly:

```swift
// CyclePhase.swift
static func estimate(cycleDay: Int, cycleLength: Int = 28) -> CyclePhase {
    let menstruationEnd = Int(Double(length) * 0.18) // ~5 days for 28
    let follicularEnd   = Int(Double(length) * 0.46) // ~13 days for 28
    let ovulationEnd    = Int(Double(length) * 0.57) // ~16 days for 28
    ...
}
```

---

## HealthKit Integration

Teko reads from a single source of truth — Apple HealthKit — which means data from Clue, Flo, Strava, and Runna is automatically available via their existing HealthKit integrations.

### Parallel Data Fetching

All three HealthKit reads happen concurrently:

```swift
// DailyScoreViewModel.swift
async let fetchRuns:  () = workoutStore.fetchRecentRuns()
async let fetchCycle: () = cycleStore.fetchMenstrualData()
async let fetchTemp:  () = cycleStore.fetchBasalTemperature()

_ = try await (fetchRuns, fetchCycle, fetchTemp)
```

### Guided App Integration Onboarding

For each third-party app (Clue, Flo, Strava, Runna), the onboarding shows a step-by-step carousel explaining how to enable HealthKit sharing from that specific app. This avoids user confusion about why health data isn't appearing.

<img width="1692" height="874" alt="image" src="https://github.com/user-attachments/assets/8733d680-e5e8-4e4e-98d1-063be9b1b014" />

### Permission Degradation

If HealthKit access is denied, the app doesn't crash or block the user — it shows a `HealthAccessBanner` explaining what's missing, and the score display is gracefully hidden. The `scoreResult.score` is set to 0 as a sentinel value.

---

## 7-Day Forecast

<img width="3840" height="1214" alt="image" src="https://github.com/user-attachments/assets/cdd0310f-b010-4b3e-9972-88f0c5b31a8e" />


**The challenge:** predicting scores for future days requires future workout data — which doesn't exist yet.

**The solution:** freeze running features at current rolling averages (a reasonable proxy for "if the user maintains their current training load"), and advance only the cycle-related features day by day.

```swift
// DailyScoreViewModel.swift
for offset in 1...7 {
    let futureCycleDay = ((baseCycleDay - 1 + offset) % cycleLength) + 1
    let futurePhase    = CyclePhase.estimate(cycleDay: futureCycleDay, cycleLength: cycleLength)

    var features = baseFeatures           // copy — running features stay the same
    features.cycleDay      = Float(futureCycleDay)
    features.phase         = Float(futurePhase.rawValue)
    features.flowIntensity = estimatedFlowIntensity(cycleDay: futureCycleDay, phase: futurePhase)
    features.basalTempDelta = estimatedBasalTempDelta(phase: futurePhase)
}
```

**Basal temperature modelling** follows the biphasic pattern: lower during follicular (−0.2°C), near-zero at ovulation, elevated during luteal (+0.3°C). This is derived directly from the physiological literature on progesterone's thermogenic effect.

---

## Apple Watch Companion

The watchOS app displays the daily score ring. The Watch does not compute the score — it receives the latest score via WatchConnectivity.

<img width="3840" height="1214" alt="image" src="https://github.com/user-attachments/assets/f366af17-1f2b-4e04-9492-f838ae28396e" />

### Why `applicationContext`?

Three WatchConnectivity APIs exist for sending data from iPhone to Watch:

| Method | Works when | Why not used |
|---|---|---|
| `sendMessage` | Both apps reachable simultaneously | Watch app may not be active when score is computed |
| `transferUserInfo` | Queues all messages for delivery | Sends all historical payloads — we only need the latest |
| **`applicationContext`** | **Delivers latest value on next app launch** | **Perfect — only the current score matters** |

`applicationContext` is persisted by the OS. Even if the Watch app is closed when the iPhone computes the score, it receives the latest value as soon as it opens.

### Data Flow

```
DailyScoreViewModel (scoreResult changes)
  └→ WatchConnectivityService.sendScore(_:date:label:)
       └→ WCSession.default.updateApplicationContext(["score": 78, "date": ..., "label": "Great"])
            └→ watchOS: WatchConnectivityReceiver.session(_:didReceiveApplicationContext:)
                 └→ Updates @Observable properties on MainActor
                      └→ WatchScoreView renders WatchScoreDialRing
```

### Cold Start Behavior

When the Watch app launches after being closed:

```swift
// WatchConnectivityReceiver.swift
func session(_ session: WCSession, activationDidCompleteWith state: WCSessionActivationState, error: Error?) {
    Task { @MainActor in
        // Read cached context from the last iPhone sync
        let context = session.receivedApplicationContext
        updateFrom(context: context)
    }
}
```

The score appears immediately on launch — no waiting for the iPhone to resend.

### Score Ring Adaptation

The iOS `ScoreDialRing` was adapted for watchOS with smaller dimensions via `@ScaledMetric`:

| Property | iOS | watchOS |
|---|---|---|
| Ring frame | ~200pt | ~100pt |
| Stroke width | 12pt | 8pt |
| Font scale base | Standard | Reduced |

---

## SwiftUI Architecture

### Modern State Management

Teko uses `@Observable` + `@MainActor` exclusively — no `ObservableObject`, `@Published`, `@StateObject`, or `@EnvironmentObject`.

```swift
@MainActor
@Observable
final class DailyScoreViewModel {
    var scoreResult: ScoreResult?
    var forecastDays: [ForecastDay] = []
    var isLoading = false
    ...
}
```

Services are passed down via `@Environment`:

```swift
// TekoApp.swift
ContentView()
    .environment(coordinator)
    .environment(viewModel)
    .environment(healthKitManager)
    .environment(purchaseService)
```

Views read them with zero boilerplate:

```swift
// DailyScoreScreen.swift
@Environment(DailyScoreViewModel.self) private var viewModel
@Environment(HealthKitManager.self) private var healthKitManager
```

### Coordinator Pattern

All navigation is driven by a single `enum Page` and a `Coordinator` class. No view knows how to navigate — it just tells the coordinator what to do next.

```swift
// Coordinator.swift
enum Page: String, Identifiable {
    case splash, start, nameInput, cycleApp, cycleInput,
         runApp, onBoarding, paywall, home, profile, ...
}

@MainActor
@Observable
final class Coordinator {
    var root: Page = .splash
    var path = NavigationPath()

    func push(_ page: Page) { path.append(page) }
    func replaceStack(with page: Page) { root = page; path = NavigationPath() }

    @ViewBuilder
    func build(page: Page) -> some View { ... }
}
```

This pattern makes the navigation graph explicit and testable in isolation from views.

### Bento-Grid Home Screen Layout

The daily score screen uses a bento-grid card layout:

```swift
// DailyScoreScreen.swift
VStack(spacing: 16) {
    ScoreDialCard(score: result.score, message: result.message)

    HStack(spacing: 16) {
        TomorrowCard(forecast: tomorrowForecast)
        PhaseCard(phase: result.phase)
    }

    ForecastCard(forecastDays: forecastDays)
}
```

<img width="3840" height="1214" alt="image" src="https://github.com/user-attachments/assets/f14f4ece-e818-4787-88b3-d370f573e05d" />


### Error & Empty States

`ContentUnavailableView` handles all error states natively:

```swift
ContentUnavailableView {
    Label(.unableToLoadScore, systemImage: "exclamationmark.triangle")
} description: {
    Text(error.localizedDescription)
} actions: {
    Button(.tryAgainButton, systemImage: "arrow.clockwise") {
        Task { await viewModel.loadTodayScore() }
    }
    .buttonStyle(.borderedProminent)
}
```

---

## Production Integrations

### Subscriptions — RevenueCat

Teko uses RevenueCat for subscription management. The integration is contained in `PurchaseService` — no RevenueCat types leak into views.

| Plan | Price | Trial |
|---|---|---|
| Monthly | $2.99/month | 2 weeks |
| Annual | $22.99/year | 2 weeks |

<img width="3840" height="1214" alt="image" src="https://github.com/user-attachments/assets/820bafa4-a1d8-4567-a2c8-061213a7ebf8" />

### Analytics — PostHog

PostHog tracks key product events without capturing any health data:

```swift
PostHogSDK.shared.capture("score_loaded", properties: [
    "score": prediction.rounded,
    "cycle_phase": phase.rawValue
])
```

Key events: `healthkit_access_granted`, `healthkit_access_denied`, `score_loaded`, `score_load_failed`, `run_session_recorded`, `model_retrained`, `cycle_app_selected`, `run_app_selected`.

### Ad Attribution — Meta Ads SDK

`MetaAdsConfiguration` handles deep link attribution via `onOpenURL`, scoped to the app delegate — no tracking touches any health or personal data.

### Privacy Manifest

`PrivacyInfo.xcprivacy` accurately declares all required-reason API usage for App Store Review. No health data types appear in the privacy nutrition label.

---

## Privacy Architecture

Privacy is not a feature — it's a structural constraint baked into every technical decision.

```
User's health data
      │
      ▼
HealthKit (on-device) ──────────────────────────── BOUNDARY: data stays here
      │
      ▼
FeatureEngineeringService (in-process)
      │
      ▼
ScoringService → RunCycleScoreUpdatable.mlmodelc (on-device inference)
      │
      ▼
MLUpdateTask → Updated model saved to app's Documents directory
      │
      ▼
SwiftData → RunSession, TrainingExample, UserProfile (on-device only)
```

No network requests carry health data. PostHog events contain only aggregate values (score, phase index) — never raw health metrics, names, or identifiers.

---

## Challenges & Lessons Learned

### 1. CoreML MultiArray vs Double Mismatch

**The bug:** The score was always 0.

**Root cause:** For updatable Neural Network Regressors with CoreML spec v4, the runtime returns the output as an `MLMultiArray` — not a scalar `Double`. The Xcode-generated Swift class calls `.doubleValue` on it, which silently returns `0.0` when the underlying type is MultiArray.

**The fix:** Bypass the generated class entirely and extract directly from the raw output:

```swift
// ScoringService.swift — correct extraction
let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["features": multiArray])
let rawOutput = try model.model.prediction(from: inputProvider)
let scoreArray = rawOutput.featureValue(for: "score")!.multiArrayValue!
let scoreValue = scoreArray[0].doubleValue
```

This cost significant debugging time. The failure was silent — the app worked, just always showed 0. The lesson: always verify CoreML output types against the model spec, not just the generated Swift interface.

---

### 2. Gradient Explosion on First Dense Layer

**The problem:** Making `dense_1` updatable caused on-device training to produce garbage scores after the first update.

**Root cause:** `dense_1` has min-max normalization folded into its weights. When SGD updates those weights using raw input values (e.g., heart rate = 158 bpm), the gradient is scaled by that raw magnitude instead of the normalized equivalent (~0.5). Result: catastrophic weight updates.

**The fix:** Mark `dense_1` as non-updatable. Only `dense_2` and `dense_output` receive gradient updates, because their inputs are post-ReLU activations bounded to [0, ∞] by construction.

This required understanding the interaction between normalization, weight initialization, and gradient flow — and then encoding that understanding directly into the CoreML model spec via Python's `coremltools`.

---

### 3. No Public Dataset for Cycle × Running

**The problem:** Training an ML model requires data. No public dataset combines menstrual cycle records with running performance metrics.

**The solution:** Generate 500 synthetic samples using an 8-layer score formula derived entirely from peer-reviewed papers. Each layer maps a specific physiological mechanism (estrogen's effect on running economy, progesterone's thermogenic effect, RPE changes during menstruation) to a score adjustment, citing the specific study.

The model ships with this base. The real personalization comes from the user's own data via `MLUpdateTask`. The synthetic dataset is the "cold start" — good enough to be useful immediately, replaced over time by individual data.

---

### 4. HealthKit Flow Intensity Off-by-One

`HKCategoryValueMenstrualFlow` raw values are `none=1, light=2, medium=3, heavy=4`. The model expects `none=0, light=1, medium=2, heavy=3`.

Using the raw HealthKit value directly introduces a silent off-by-one in every prediction. The fix is a simple remap in `CycleTrackingStore.fetchMenstrualData()`, but this kind of silent semantic mismatch between an SDK's encoding and a model's expected encoding is easy to miss and hard to detect from outputs alone.

**The lesson:** document your feature-to-HealthKit mapping explicitly, and test edge cases (no flow, heavy flow) with known expected scores.

---

### 5. WatchConnectivity Reachability is Unreliable

Initial implementation used `WCSession.sendMessage` — the Watch would receive the score when both apps were active. In practice, the Watch app was almost never active when the iPhone computed the score (which happens in the background or immediately on app launch).

**The switch to `applicationContext`:** the OS guarantees delivery of the most recent context the next time the Watch app becomes active. This made score delivery reliable regardless of whether the Watch app was running when the iPhone computed the score.

**The additional win:** `applicationContext` is persisted across Watch app relaunches. The score is available immediately on cold start without any network round-trip.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Swift 6.2 |
| UI | SwiftUI (`@Observable`, `NavigationStack`, `@Environment`) |
| State | `@Observable` + `@MainActor` |
| Persistence | SwiftData |
| Machine Learning | Core ML (`MLUpdateTask`, `NeuralNetworkBuilder`) |
| ML Training | PyTorch + coremltools (Python) |
| Health Data | HealthKit |
| Watch Connectivity | WatchConnectivity (`applicationContext`) |
| Subscriptions | RevenueCat |
| Analytics | PostHog |
| Ad Attribution | Meta Ads SDK |
| Deployment | iOS 18.6+, watchOS |
| Architecture | MVVM + Coordinator |

---

## Contact

**Eduardo Bertol** — iOS Engineer
[eduardobbertol@gmail.com](mailto:eduardobbertol@gmail.com)

---

*Built at Apple Developer Academy · Production app on the App Store · All health data stays on-device.*
