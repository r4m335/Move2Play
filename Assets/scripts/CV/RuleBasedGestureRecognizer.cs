using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class RuleBasedGestureRecognizer
{
    // ---------- Configuration ----------
    private const float SMOOTHING_WINDOW_MS = 100f;
    private const float JUMP_THRESHOLD = 0.10f;
    private const float RUN_ABD_THRESH = 0.8f;
    private const float LEAN_ABD_THRESH = 0.50f;
    private const float HEIGHT_MARGIN = 0.8f;
    private const float HANDS_UP_THRESH = 0.8f;
    private const float PROX_THRESH = 0.35f;
    private const float KNEE_BENT_THRESH = 145f;
    private const float ELEV_THRESH = 0.25f;
    private const float FWD_THRESH = -0.6f;
    private const float LEAN_LOCKOUT_AFTER_PUNCH = 0.8f;

    private const int RUN_EXIT_THRESHOLD = 1;
    private const int BUFFER_CAPACITY = 30;

    // Onset detection for jump and kick (prevents multiple triggers per motion)
    private bool wasJumpTriggeredLastFrame = false;
    private bool wasKickTriggeredLastFrame = false;
    private float lastJumpTriggerTime = -1f;
    private float lastKickTriggerTime = -1f;
    private const float JUMP_RETRIGGER_DELAY = 0.5f;   // seconds
    private const float KICK_RETRIGGER_DELAY = 0.5f;

    private static readonly Dictionary<string, float> DEFAULT_COOLDOWNS = new Dictionary<string, float>
    {
        { "attack", 400f }, { "punch", 300f }, { "kick", 600f }, { "dodge", 400f }, { "block", 300f }, { "jump", 800f }
    };

    private static readonly Dictionary<string, (string action, Dictionary<string, object> parameters)> GestureActionMap =
        new Dictionary<string, (string, Dictionary<string, object>)>
        {
            { "idle", ("no_action", new Dictionary<string, object> { { "description", "standing still" } }) },
            { "block", ("block", new Dictionary<string, object> { { "damage_reduction", 0.7f }, { "duration", 2f } }) },
            { "kick", ("kick", new Dictionary<string, object> { { "damage", 35 } }) },
            { "lean_left", ("lean_left", new Dictionary<string, object> { { "direction", "left" } }) },
            { "lean_right", ("lean_right", new Dictionary<string, object> { { "direction", "right" } }) },
            { "punch", ("punch", new Dictionary<string, object> { { "damage", 25 } }) },
            { "run", ("forward_movement", new Dictionary<string, object> { { "speed", 1f } }) },
            { "squat", ("dodge", new Dictionary<string, object> { { "duration", 0.5f } }) },
            { "jump", ("jump", new Dictionary<string, object> { { "height", 1f } }) }
        };

    // ---------- Internal State ----------
    private Queue<List<PoseLandmark>> rawBuffer = new Queue<List<PoseLandmark>>();
    private ActionCooldownManager cooldownManager;
    private PredictionSmoother smoother;
    private GestureState state;
    private float lastProcessTime;

    // Landmark indices
    private static readonly int[] HIPS = { 23, 24 };
    private const int NOSE = 0;
    private const int LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
    private const int LEFT_ELBOW = 13, RIGHT_ELBOW = 14;
    private const int LEFT_WRIST = 15, RIGHT_WRIST = 16;
    private const int LEFT_HIP = 23, RIGHT_HIP = 24;
    private const int LEFT_KNEE = 25, RIGHT_KNEE = 26;
    private const int LEFT_ANKLE = 27, RIGHT_ANKLE = 28;

    private FrameResult lastFrameResult;

    public RuleBasedGestureRecognizer()
    {
        cooldownManager = new ActionCooldownManager(DEFAULT_COOLDOWNS);
        smoother = new PredictionSmoother(SMOOTHING_WINDOW_MS);
        state = new GestureState();
        lastProcessTime = Time.time;
    }

    public FrameResult GetLastResult() => lastFrameResult;

    public GestureResult ProcessFrame(List<PoseLandmark> landmarks, float deltaTime)
    {
        float now = Time.time;
        if (now - lastProcessTime < 1f / 30f) return null;
        lastProcessTime = now;

        if (landmarks == null || landmarks.Count < 33)
            return new GestureResult { gesture = null, skipReason = "no_pose" };

        rawBuffer.Enqueue(new List<PoseLandmark>(landmarks));
        while (rawBuffer.Count > BUFFER_CAPACITY) rawBuffer.Dequeue();

        float legMotion = EstimateSegmentMotion(rawBuffer.ToList(), true);
        float jumpImpulse = EstimateJumpImpulse(rawBuffer.ToList());
        float legAsymmetry = EstimateLegAsymmetry(rawBuffer.ToList());
        var (lAbd, rAbd, lHeight, rHeight) = EstimateArmAbduction(rawBuffer.ToList());
        var (lKnee, rKnee) = EstimateKneeBending(rawBuffer.ToList());
        var (lElev, rElev) = EstimateLegElevation(rawBuffer.ToList());
        var (lProx, rProx) = EstimateHandToHeadProximity(rawBuffer.ToList());
        var (lFwd, rFwd) = EstimateHandForwardExtension(rawBuffer.ToList());

        bool hipsVisible = AreJointsVisible(landmarks, HIPS, 0.35f);
        bool legsVisible = AreJointsVisible(landmarks, new[] { LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE }, 0.30f);

        bool bothHandsWide = (lAbd >= RUN_ABD_THRESH && rAbd >= RUN_ABD_THRESH &&
                              Mathf.Abs(lHeight) < HEIGHT_MARGIN && Mathf.Abs(rHeight) < HEIGHT_MARGIN &&
                              !(lProx < PROX_THRESH && rProx < PROX_THRESH) && !(lHeight > HANDS_UP_THRESH && rHeight > HANDS_UP_THRESH));

        bool leftOnlyWide = (lAbd >= LEAN_ABD_THRESH && lAbd > (rAbd + 0.20f) &&
                             Mathf.Abs(lHeight) < HEIGHT_MARGIN && !(lProx < PROX_THRESH && rProx < PROX_THRESH) &&
                             !(lHeight > HANDS_UP_THRESH && rHeight > HANDS_UP_THRESH) && !bothHandsWide);

        bool rightOnlyWide = (rAbd >= LEAN_ABD_THRESH && rAbd > (lAbd + 0.20f) &&
                              Mathf.Abs(rHeight) < HEIGHT_MARGIN && !(lProx < PROX_THRESH && rProx < PROX_THRESH) &&
                              !(lHeight > HANDS_UP_THRESH && rHeight > HANDS_UP_THRESH) && !bothHandsWide);

        bool bothKneesBent = (lKnee < KNEE_BENT_THRESH && rKnee < KNEE_BENT_THRESH);
        bool legRaised = (lElev > ELEV_THRESH || rElev > ELEV_THRESH);
        bool handsNearHead = (lProx < PROX_THRESH && rProx < PROX_THRESH);
        bool handsUp = (lHeight > HANDS_UP_THRESH && rHeight > HANDS_UP_THRESH);

        bool handsCrossed = false;
        if (landmarks.Count > RIGHT_WRIST)
            handsCrossed = (landmarks[LEFT_WRIST].position.x < landmarks[RIGHT_WRIST].position.x);

        bool lIsFwd = (lFwd < FWD_THRESH && lHeight > -0.5f && !handsCrossed);
        bool rIsFwd = (rFwd < FWD_THRESH && rHeight > -0.5f && !handsCrossed);
        bool handsFwd = lIsFwd || rIsFwd;

        bool kickLegTrigger = ((legsVisible && legMotion >= 0.08f && legAsymmetry >= 0.50f) || legRaised) && jumpImpulse < JUMP_THRESHOLD;

        string selectedGesture = "idle";
        float confidence = 0f;

        // ----- Jump detection (onset + delay) -----
        bool forceJump = false;
        if (hipsVisible && jumpImpulse >= JUMP_THRESHOLD)
        {
            state.jumpConsecFrames++;
            if (state.jumpConsecFrames >= 2 || jumpImpulse > 0.35f) forceJump = true;
        }
        else state.jumpConsecFrames = 0;

        bool jumpNow = forceJump && !wasJumpTriggeredLastFrame && (now - lastJumpTriggerTime) >= JUMP_RETRIGGER_DELAY;
        if (jumpNow)
        {
            selectedGesture = "jump";
            lastJumpTriggerTime = now;
        }
        wasJumpTriggeredLastFrame = forceJump;

        // ----- Kick detection (onset + delay) -----
        bool kickNow = kickLegTrigger && !wasKickTriggeredLastFrame && (now - lastKickTriggerTime) >= KICK_RETRIGGER_DELAY;
        if (kickNow)
        {
            selectedGesture = "kick";
            lastKickTriggerTime = now;
        }
        wasKickTriggeredLastFrame = kickLegTrigger;

        // ----- Other gestures (run, lean, etc.) -----
        if (selectedGesture == "idle")
        {
            if (bothHandsWide) selectedGesture = "run";
            else if (leftOnlyWide && (now - state.lastPunchTime) >= LEAN_LOCKOUT_AFTER_PUNCH) selectedGesture = "lean_right";
            else if (rightOnlyWide && (now - state.lastPunchTime) >= LEAN_LOCKOUT_AFTER_PUNCH) selectedGesture = "lean_left";
            else if (handsNearHead) selectedGesture = "block";
            else if (handsFwd) selectedGesture = "punch";
            else if (bothKneesBent) selectedGesture = "squat";
            else selectedGesture = "idle";
        }

        confidence = 0.85f;

        smoother.AddPrediction(selectedGesture, confidence);
        var smoothed = smoother.GetSmoothedPrediction();
        if (smoothed != null)
        {
            selectedGesture = smoothed.Value.gesture;
            confidence = smoothed.Value.confidence;
        }

        var persistentGestures = new[] { "run", "jump", "lean_left", "lean_right" };
        if (persistentGestures.Contains(selectedGesture)) state.runExitCounter = 0;
        else state.runExitCounter++;

        bool shouldStopRun = state.runActive && !persistentGestures.Contains(selectedGesture) && state.runExitCounter >= RUN_EXIT_THRESHOLD;

        string actionName = null;
        bool cooldownBlocked = false;
        Dictionary<string, object> actionParams = null;

        if (shouldStopRun)
        {
            actionName = "run_stop";
            state.runActive = false;
        }
        else if (selectedGesture == "run" && !state.runActive)
        {
            actionName = "run_start";
            state.runActive = true;
        }
        else if (selectedGesture == "block" && !state.blockActive)
        {
            actionName = "block_start";
            state.blockActive = true;
        }
        else if (state.blockActive && selectedGesture != "block")
        {
            actionName = "block_stop";
            state.blockActive = false;
        }
        else if (selectedGesture != "idle" && !persistentGestures.Contains(selectedGesture))
        {
            if (GestureActionMap.TryGetValue(selectedGesture, out var mapping))
            {
                actionName = mapping.action;
                actionParams = mapping.parameters;
                if (!cooldownManager.CanPerformAction(actionName)) cooldownBlocked = true;
                else cooldownManager.RecordAction(actionName);
            }
        }

        if (selectedGesture == "punch") state.lastPunchTime = Time.time;

        state.currentGesture = selectedGesture;
        state.currentConfidence = confidence;
        state.currentAction = actionName;
        state.prevGesture = selectedGesture;

        // Store for UI
        float[] allProbs = new float[9];
        string[] allGestures = new string[] { "idle", "run", "punch", "kick", "block", "lean_left", "lean_right", "squat", "jump" };
        for (int i = 0; i < allGestures.Length; i++)
            allProbs[i] = (allGestures[i] == selectedGesture) ? confidence : 0.05f;

        lastFrameResult = new FrameResult
        {
            gesture = selectedGesture,
            confidence = confidence,
            allProbabilities = allProbs,
            legMotion = legMotion,
            jumpImpulse = jumpImpulse,
            legAsymmetry = legAsymmetry,
            leftAbduction = lAbd,
            rightAbduction = rAbd
        };

        return new GestureResult
        {
            gesture = selectedGesture,
            confidence = confidence,
            action = cooldownBlocked ? null : actionName,
            actionParams = actionParams,
            cooldownBlocked = cooldownBlocked,
            skipReason = null
        };
    }

    // ---------- Helper methods ----------
    private float EstimateSegmentMotion(List<List<PoseLandmark>> buffer, bool legs)
    {
        if (buffer.Count < 2) return 0f;
        int[] indices = legs ? new[] { LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE }
                             : new[] { LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST };
        float total = 0f; int count = 0;
        for (int i = 1; i < buffer.Count; i++)
        {
            var prev = buffer[i - 1];
            var curr = buffer[i];
            foreach (int idx in indices)
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    total += Vector3.Distance(prev[idx].position, curr[idx].position);
                    count++;
                }
            }
        }
        return count > 0 ? total / count : 0f;
    }

    private float EstimateJumpImpulse(List<List<PoseLandmark>> buffer)
    {
        if (buffer.Count < 3) return 0f;
        List<float> hipY = new List<float>();
        foreach (var frame in buffer)
        {
            float y = 0f; int valid = 0;
            foreach (int hip in HIPS)
            {
                if (hip < frame.Count && frame[hip].visibility > 0.35f)
                {
                    y += frame[hip].position.y;
                    valid++;
                }
            }
            if (valid > 0) hipY.Add(y / valid);
        }
        if (hipY.Count < 3) return 0f;
        float range = hipY.Max() - hipY.Min();
        float maxDiff = 0f;
        for (int i = 1; i < hipY.Count; i++)
            maxDiff = Mathf.Max(maxDiff, Mathf.Abs(hipY[i] - hipY[i-1]));
        return 0.6f * range + 0.4f * maxDiff;
    }

    private (float left, float right, float leftHeight, float rightHeight) EstimateArmAbduction(List<List<PoseLandmark>> buffer)
    {
        float lAbd = 0f, rAbd = 0f, lH = 0f, rH = 0f;
        int count = 0;
        foreach (var frame in buffer)
        {
            if (frame.Count <= RIGHT_SHOULDER) continue;
            float visL = frame[LEFT_SHOULDER].visibility;
            float visR = frame[RIGHT_SHOULDER].visibility;
            if (visL < 0.35f || visR < 0.35f) continue;

            Vector3 lSh = frame[LEFT_SHOULDER].position;
            Vector3 rSh = frame[RIGHT_SHOULDER].position;
            float centerX = (lSh.x + rSh.x) / 2f;
            float shoulderWidth = Mathf.Abs(lSh.x - rSh.x);
            float torsoLen = Mathf.Abs(lSh.y - frame[LEFT_HIP].position.y);

            lAbd += Mathf.Abs(frame[LEFT_WRIST].position.x - centerX) / Mathf.Max(shoulderWidth, 0.01f);
            rAbd += Mathf.Abs(frame[RIGHT_WRIST].position.x - centerX) / Mathf.Max(shoulderWidth, 0.01f);
            lH += (lSh.y - frame[LEFT_WRIST].position.y) / Mathf.Max(torsoLen, 0.1f);
            rH += (rSh.y - frame[RIGHT_WRIST].position.y) / Mathf.Max(torsoLen, 0.1f);
            count++;
        }
        if (count == 0) return (0,0,0,0);
        return (lAbd/count, rAbd/count, lH/count, rH/count);
    }

    private (float left, float right) EstimateKneeBending(List<List<PoseLandmark>> buffer)
    {
        float lAngle = 180f, rAngle = 180f;
        int lCount = 0, rCount = 0;
        foreach (var frame in buffer)
        {
            if (frame.Count > LEFT_ANKLE &&
                frame[LEFT_HIP].visibility > 0.3f &&
                frame[LEFT_KNEE].visibility > 0.3f &&
                frame[LEFT_ANKLE].visibility > 0.3f)
            {
                Vector2 hip = new Vector2(frame[LEFT_HIP].position.x, frame[LEFT_HIP].position.y);
                Vector2 knee = new Vector2(frame[LEFT_KNEE].position.x, frame[LEFT_KNEE].position.y);
                Vector2 ankle = new Vector2(frame[LEFT_ANKLE].position.x, frame[LEFT_ANKLE].position.y);
                lAngle += AngleBetween2D(hip, knee, ankle);
                lCount++;
            }

            if (frame.Count > RIGHT_ANKLE &&
                frame[RIGHT_HIP].visibility > 0.3f &&
                frame[RIGHT_KNEE].visibility > 0.3f &&
                frame[RIGHT_ANKLE].visibility > 0.3f)
            {
                Vector2 hip = new Vector2(frame[RIGHT_HIP].position.x, frame[RIGHT_HIP].position.y);
                Vector2 knee = new Vector2(frame[RIGHT_KNEE].position.x, frame[RIGHT_KNEE].position.y);
                Vector2 ankle = new Vector2(frame[RIGHT_ANKLE].position.x, frame[RIGHT_ANKLE].position.y);
                rAngle += AngleBetween2D(hip, knee, ankle);
                rCount++;
            }
        }
        return (lCount > 0 ? lAngle / lCount : 180f, rCount > 0 ? rAngle / rCount : 180f);
    }

    private (float left, float right) EstimateLegElevation(List<List<PoseLandmark>> buffer)
    {
        float lElev = 0f, rElev = 0f;
        int count = 0;
        foreach (var frame in buffer)
        {
            if (frame.Count <= RIGHT_ANKLE) continue;
            float torso = Mathf.Abs(frame[LEFT_SHOULDER].position.y - frame[LEFT_HIP].position.y);
            if (torso < 0.01f) continue;

            bool lVis = frame[LEFT_ANKLE].visibility > 0.3f;
            bool rVis = frame[RIGHT_ANKLE].visibility > 0.3f;
            if (lVis && rVis)
            {
                lElev += (frame[RIGHT_ANKLE].position.y - frame[LEFT_ANKLE].position.y) / torso;
                rElev += (frame[LEFT_ANKLE].position.y - frame[RIGHT_ANKLE].position.y) / torso;
                count++;
            }
            else if (lVis)
            {
                lElev += (1f - frame[LEFT_ANKLE].position.y) / torso;
                count++;
            }
            else if (rVis)
            {
                rElev += (1f - frame[RIGHT_ANKLE].position.y) / torso;
                count++;
            }
        }
        return (count > 0 ? lElev/count : 0f, count > 0 ? rElev/count : 0f);
    }

    private (float left, float right) EstimateHandToHeadProximity(List<List<PoseLandmark>> buffer)
    {
        float lProx = 1f, rProx = 1f;
        int lCount = 0, rCount = 0;
        foreach (var frame in buffer)
        {
            if (frame.Count <= NOSE || frame[NOSE].visibility < 0.3f) continue;
            Vector2 head = new Vector2(frame[NOSE].position.x, frame[NOSE].position.y);
            float torso = Mathf.Abs(frame[LEFT_SHOULDER].position.y - frame[LEFT_HIP].position.y);
            if (torso < 0.01f) torso = 0.1f;

            if (frame.Count > LEFT_WRIST && frame[LEFT_WRIST].visibility > 0.3f)
            {
                Vector2 wrist = new Vector2(frame[LEFT_WRIST].position.x, frame[LEFT_WRIST].position.y);
                float dist = Vector2.Distance(wrist, head) / torso;
                lProx += dist;
                lCount++;
            }
            if (frame.Count > RIGHT_WRIST && frame[RIGHT_WRIST].visibility > 0.3f)
            {
                Vector2 wrist = new Vector2(frame[RIGHT_WRIST].position.x, frame[RIGHT_WRIST].position.y);
                float dist = Vector2.Distance(wrist, head) / torso;
                rProx += dist;
                rCount++;
            }
        }
        return (lCount > 0 ? lProx / lCount : 1f, rCount > 0 ? rProx / rCount : 1f);
    }

    private (float left, float right) EstimateHandForwardExtension(List<List<PoseLandmark>> buffer)
    {
        float lExt = 0f, rExt = 0f;
        int lCount = 0, rCount = 0;
        foreach (var frame in buffer)
        {
            if (frame.Count <= RIGHT_HIP) continue;
            Vector3 hipCenter = (frame[LEFT_HIP].position + frame[RIGHT_HIP].position) / 2f;
            float torso = Mathf.Abs(frame[LEFT_SHOULDER].position.y - frame[LEFT_HIP].position.y);
            if (frame.Count > LEFT_WRIST && frame[LEFT_WRIST].visibility > 0.3f)
            {
                lExt += (frame[LEFT_WRIST].position.z - hipCenter.z) / torso;
                lCount++;
            }
            if (frame.Count > RIGHT_WRIST && frame[RIGHT_WRIST].visibility > 0.3f)
            {
                rExt += (frame[RIGHT_WRIST].position.z - hipCenter.z) / torso;
                rCount++;
            }
        }
        return (lCount > 0 ? lExt/lCount : 0f, rCount > 0 ? rExt/rCount : 0f);
    }

    private float EstimateLegAsymmetry(List<List<PoseLandmark>> buffer)
    {
        if (buffer.Count < 2) return 0f;
        float leftMotion = 0f, rightMotion = 0f;
        int leftCount = 0, rightCount = 0;
        for (int i = 1; i < buffer.Count; i++)
        {
            var prev = buffer[i-1];
            var curr = buffer[i];
            foreach (int idx in new[] { LEFT_HIP, LEFT_KNEE, LEFT_ANKLE })
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    leftMotion += Vector3.Distance(prev[idx].position, curr[idx].position);
                    leftCount++;
                }
            }
            foreach (int idx in new[] { RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE })
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    rightMotion += Vector3.Distance(prev[idx].position, curr[idx].position);
                    rightCount++;
                }
            }
        }
        leftMotion = leftCount > 0 ? leftMotion / leftCount : 0f;
        rightMotion = rightCount > 0 ? rightMotion / rightCount : 0f;
        float total = leftMotion + rightMotion + 1e-6f;
        return Mathf.Abs(leftMotion - rightMotion) / total;
    }

    private bool AreJointsVisible(List<PoseLandmark> frame, int[] indices, float threshold)
    {
        foreach (int i in indices)
            if (i < frame.Count && frame[i].visibility < threshold) return false;
        return true;
    }

    private float AngleBetween2D(Vector2 a, Vector2 b, Vector2 c)
    {
        Vector2 ba = a - b;
        Vector2 bc = c - b;
        float dot = Vector2.Dot(ba, bc);
        float mag = ba.magnitude * bc.magnitude;
        if (mag < 1e-6f) return 180f;
        return Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(dot / mag, -1f, 1f));
    }

    private float AngleBetween(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ba = a - b;
        Vector3 bc = c - b;
        float dot = Vector3.Dot(ba, bc);
        float mag = ba.magnitude * bc.magnitude;
        if (mag < 1e-6f) return 180f;
        return Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(dot / mag, -1f, 1f));
    }

    // ---------- Inner classes ----------
    private class ActionCooldownManager
    {
        private Dictionary<string, float> cooldowns;
        private Dictionary<string, float> lastActionTimes;

        public ActionCooldownManager(Dictionary<string, float> cooldownMap)
        {
            cooldowns = new Dictionary<string, float>(cooldownMap);
            lastActionTimes = new Dictionary<string, float>();
        }

        public bool CanPerformAction(string action)
        {
            if (!cooldowns.ContainsKey(action)) return true;
            float now = Time.time * 1000f;
            float last = lastActionTimes.GetValueOrDefault(action, 0f);
            return (now - last) >= cooldowns[action];
        }

        public void RecordAction(string action)
        {
            if (cooldowns.ContainsKey(action))
                lastActionTimes[action] = Time.time * 1000f;
        }
    }

    private class PredictionSmoother
    {
        private struct Prediction { public float timestampMs; public string gesture; public float confidence; }
        private List<Prediction> history = new List<Prediction>();
        private float windowMs;

        public PredictionSmoother(float windowMs) { this.windowMs = windowMs; }

        public void AddPrediction(string gesture, float confidence)
        {
            history.Add(new Prediction { timestampMs = Time.time * 1000f, gesture = gesture, confidence = confidence });
            PruneOld();
        }

        public (string gesture, float confidence)? GetSmoothedPrediction()
        {
            PruneOld();
            if (history.Count == 0) return null;
            Dictionary<string, float> votes = new Dictionary<string, float>();
            float totalWeight = 0f;
            float now = Time.time * 1000f;
            foreach (var p in history)
            {
                float age = now - p.timestampMs;
                if (age > windowMs) continue;
                float timeWeight = 1f - (age / windowMs);
                float weight = timeWeight * p.confidence;
                votes[p.gesture] = votes.GetValueOrDefault(p.gesture, 0f) + weight;
                totalWeight += weight;
            }
            if (totalWeight == 0) return null;
            var best = votes.OrderByDescending(kv => kv.Value).First();
            return (best.Key, best.Value / totalWeight);
        }

        private void PruneOld()
        {
            float cutoff = Time.time * 1000f - windowMs;
            history.RemoveAll(p => p.timestampMs < cutoff);
        }
    }
}

public class FrameResult
{
    public string gesture;
    public float confidence;
    public float[] allProbabilities;
    public float legMotion;
    public float jumpImpulse;
    public float legAsymmetry;
    public float leftAbduction;
    public float rightAbduction;
}

public class GestureResult
{
    public string gesture;
    public float confidence;
    public string action;
    public Dictionary<string, object> actionParams;
    public bool cooldownBlocked;
    public string skipReason;
}