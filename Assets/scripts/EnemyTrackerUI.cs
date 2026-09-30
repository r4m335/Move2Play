using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class EnemyTrackerUI : MonoBehaviour
{
    [Header("References")]
    public Transform player;
    public RectTransform arrow;
    public TMP_Text distanceText;

    [Header("Settings")]
    public float hideDistance = 10f;

    [Header("Audio")]
    public AudioSource heartbeatSource;
    public float maxHeartbeatDistance = 20f;   // Heartbeat becomes audible at this distance
    public float fullVolumeDistance = 5f;      // Below this distance, heartbeat is at full volume & max pitch

    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;

        if (player == null)
        {
            GameObject p = GameObject.FindGameObjectWithTag("Player");
            if (p != null)
                player = p.transform;
        }

        // Start the heartbeat loop silently – we only control volume/pitch from now on
        if (heartbeatSource != null)
        {
            heartbeatSource.loop = true;
            heartbeatSource.volume = 0f;
            heartbeatSource.Play();
        }
    }

    void Update()
    {
        // Immediately mute heartbeat (and skip UI updates) if the game has ended
        if (AudioManager.Instance != null && AudioManager.Instance.gameEnded)
        {
            if (heartbeatSource != null)
                heartbeatSource.volume = 0f;

            return;
        }

        if (player == null)
            return;

        GameObject nearestEnemy = FindNearestEnemy();

        if (nearestEnemy == null)
        {
            arrow.gameObject.SetActive(false);
            distanceText.gameObject.SetActive(false);

            // Mute heartbeat when no enemies exist
            if (heartbeatSource != null)
                heartbeatSource.volume = 0f;

            return;
        }

        float distance = Vector3.Distance(player.position, nearestEnemy.transform.position);

        // --- HEARTBEAT AUDIO (always playing, volume/pitch smoothly controlled) ---
        if (heartbeatSource != null)
        {
            if (distance <= fullVolumeDistance)
            {
                // Enemy very close – full intensity
                heartbeatSource.volume = 1f;
                heartbeatSource.pitch = 1.5f;
            }
            else if (distance <= maxHeartbeatDistance)
            {
                // Linear fade from full intensity (at fullVolumeDistance) to silent (at maxHeartbeatDistance)
                float t = 1f - Mathf.InverseLerp(fullVolumeDistance, maxHeartbeatDistance, distance);

                heartbeatSource.volume = t;
                heartbeatSource.pitch = Mathf.Lerp(1f, 1.5f, t);
            }
            else
            {
                // Too far – silent
                heartbeatSource.volume = 0f;
            }
        }
        // --------------------------------------------------------------

        // Arrow color & distance text
        Color uiColor;

        if (distance > 40f)
        {
            uiColor = Color.white;
        }
        else if (distance > 20f)
        {
            uiColor = Color.yellow;
        }
        else if (distance > 10f)
        {
            uiColor = new Color(1f, 0.5f, 0f); // Orange
        }
        else
        {
            uiColor = Color.red;
        }

        Image arrowImage = arrow.GetComponent<Image>();
        if (arrowImage != null)
            arrowImage.color = uiColor;

        distanceText.color = uiColor;

        // Arrow visibility – hide when close, but heartbeat continues
        if (distance < hideDistance)
        {
            arrow.gameObject.SetActive(false);
            distanceText.gameObject.SetActive(false);
        }
        else
        {
            arrow.gameObject.SetActive(true);
            distanceText.gameObject.SetActive(true);

            distanceText.text = Mathf.RoundToInt(distance) + "m";
            RotateArrow(nearestEnemy.transform);
        }
    }

    GameObject FindNearestEnemy()
    {
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        GameObject nearest = null;
        float closestDistance = Mathf.Infinity;

        foreach (GameObject enemy in enemies)
        {
            float d = Vector3.Distance(player.position, enemy.transform.position);
            if (d < closestDistance)
            {
                closestDistance = d;
                nearest = enemy;
            }
        }

        return nearest;
    }

    void RotateArrow(Transform enemy)
    {
        Vector3 dir = enemy.position - player.position;
        dir.y = 0;

        float angle = Vector3.SignedAngle(player.forward, dir, Vector3.up);
        arrow.localRotation = Quaternion.Euler(0, 0, -angle);
    }
}