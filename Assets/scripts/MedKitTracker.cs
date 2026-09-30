using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class MedKitTracker : MonoBehaviour
{
    [Header("References")]
    public Transform player;
    public RectTransform arrow;          // The arrow UI element
    public TMP_Text distanceText;        // Distance label
    public GameObject trackerRoot;       // Parent object to show/hide
    public PlayerStats playerStats;

    [Header("Settings")]
    public float lowHealthPercent = 0.3f;   // Show tracker when health ≤ 30%
    public float updateInterval = 0.3f;     // How often to search for the nearest medkit

    [Header("Pulse Settings")]
    public float pulseSpeed = 4f;
    public float pulseMinScale = 0.9f;
    public float pulseMaxScale = 1.1f;
    public float criticalHealthPercent = 0.1f; // Pulse only when health ≤ 10%

    private Camera mainCamera;
    private Transform nearestMedKit;
    private Image arrowImage;
    private float searchTimer;
    private GameObject[] allMedKits;       // Cached array

    void Start()
    {
        mainCamera = Camera.main;

        // Cache components
        if (arrow != null)
            arrowImage = arrow.GetComponent<Image>();
        if (distanceText == null)
            Debug.LogWarning("DistanceText not assigned on MedKitTracker.");
        if (trackerRoot == null)
            Debug.LogWarning("TrackerRoot not assigned on MedKitTracker.");

        // Cache all medkits once (they rarely change)
        RefreshMedKitList();

        // Start with tracker hidden
        if (trackerRoot != null)
            trackerRoot.SetActive(false);
    }

    void Update()
    {
        // Safety checks
        if (playerStats == null || player == null || trackerRoot == null)
            return;

        float healthPercent = playerStats.currentHealth / playerStats.maxHealth;

        // Hide tracker if health is above the threshold
        if (healthPercent > lowHealthPercent)
        {
            trackerRoot.SetActive(false);
            return;
        }

        // Show tracker
        trackerRoot.SetActive(true);

        // Refresh medkit list periodically (in case some are destroyed/spawned)
        searchTimer += Time.deltaTime;
        if (searchTimer >= updateInterval)
        {
            searchTimer = 0f;
            RefreshMedKitList();
            FindNearestMedKit();
        }

        // If no medkit found, hide tracker
        if (nearestMedKit == null)
        {
            trackerRoot.SetActive(false);
            return;
        }

        // ----- Direction & Distance -----
        Vector3 dir = nearestMedKit.position - player.position;
        dir.y = 0; // ignore height for 2D compass

        float angle = Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg;
        arrow.rotation = Quaternion.Euler(0, 0, -angle + player.eulerAngles.y);

        float distance = Vector3.Distance(player.position, nearestMedKit.position);

        // ----- High‑contrast color logic -----
        Color uiColor;
        if (distance > 100f)
            uiColor = Color.white;
        else if (distance > 50f)
            uiColor = Color.yellow;
        else if (distance > 25f)
            uiColor = new Color(1f, 0.65f, 0f); // Orange
        else if (distance > 10f)
            uiColor = Color.red;
        else
            uiColor = new Color(1f, 0f, 1f);   // Magenta

        // Apply color to arrow and text
        if (arrowImage != null)
            arrowImage.color = uiColor;
        if (distanceText != null)
            distanceText.color = uiColor;

        // ----- Distance text -----
        if (distanceText != null)
        {
            if (distance < 10f)
                distanceText.text = distance.ToString("F1") + "m"; // one decimal for close range
            else
                distanceText.text = Mathf.RoundToInt(distance) + "m";
        }

        // ----- Pulsing when critically low (≤ 10%) -----
        if (healthPercent <= criticalHealthPercent && trackerRoot.activeSelf)
        {
            float pulse = Mathf.Lerp(pulseMinScale, pulseMaxScale,
                Mathf.Sin(Time.time * pulseSpeed) * 0.5f + 0.5f);
            trackerRoot.transform.localScale = Vector3.one * pulse;
        }
        else
        {
            trackerRoot.transform.localScale = Vector3.one;
        }
    }

    // Refresh the list of all medkits in the scene
    void RefreshMedKitList()
    {
        allMedKits = GameObject.FindGameObjectsWithTag("MedKit");
    }

    // Find the closest medkit from the cached list
    void FindNearestMedKit()
    {
        float closestDistance = Mathf.Infinity;
        nearestMedKit = null;

        foreach (GameObject medkit in allMedKits)
        {
            if (medkit == null) continue; // skip destroyed ones
            float d = Vector3.Distance(player.position, medkit.transform.position);
            if (d < closestDistance)
            {
                closestDistance = d;
                nearestMedKit = medkit.transform;
            }
        }
    }
}