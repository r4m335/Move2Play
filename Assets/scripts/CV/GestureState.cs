public class GestureState
{
    public string currentGesture = "idle";
    public float currentConfidence = 0f;
    public string currentAction = null;
    public bool runActive = false;
    public bool blockActive = false;
    public string prevGesture = "idle";
    public int runExitCounter = 0;
    public int jumpConsecFrames = 0;
    public float lastPunchTime = 0f;
}