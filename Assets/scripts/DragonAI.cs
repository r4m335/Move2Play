using UnityEngine;
using System.Collections;

public class DragonAI : MonoBehaviour
{
    [Header("Waypoints")]
    public Transform[] flyPoints;
    public Transform[] landPoints;

    [Header("Movement Settings")]
    public float flySpeed = 12f;
    public float landingFlySpeed = 4f;        // slower, cinematic descent
    public float rotateSpeed = 1.5f;
    public float pointReachDistance = 4f;

    [Header("Behaviour")]
    public float landChance = 0.25f;
    public float idleTime = 8f;

    private Animator anim;
    private Transform currentTarget;

    private bool isFlying = false;
    private bool isTransitioning = false;
    private float idleTimer = 0f;

    void Start()
    {
        anim = GetComponent<Animator>();

        // Safety check
        if (flyPoints.Length == 0)
        {
            Debug.LogError("DragonAI: No fly points assigned!");
            enabled = false;
            return;
        }

        ChooseRandomFlyPoint();
        StartCoroutine(TakeOffRoutine());
    }

    void Update()
    {
        if (isTransitioning) return;

        if (isFlying)
            FlyMovement();
        else
            GroundIdle();
    }

    // --------------------------------------------------
    // Flying movement – smoothed target for both rotation and movement
    // --------------------------------------------------
    void FlyMovement()
    {
        if (currentTarget == null) return;

        // Smooth altitude: blend target Y with current Y
        Vector3 targetPos = currentTarget.position;
        targetPos.y = Mathf.Lerp(transform.position.y, currentTarget.position.y, 0.3f);

        Vector3 direction = (targetPos - transform.position).normalized;

        if (direction != Vector3.zero)
        {
            Quaternion lookRot = Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(transform.rotation, lookRot, rotateSpeed * Time.deltaTime);
        }

        // Move toward the smoothed target – rotation and movement now match
        transform.position = Vector3.MoveTowards(
            transform.position,
            targetPos,
            flySpeed * Time.deltaTime
        );

        // Check proximity to the raw point (so altitude doesn't prevent arrival)
        if (Vector3.Distance(transform.position, currentTarget.position) < pointReachDistance)
        {
            if (Random.value < landChance)
                StartCoroutine(LandRoutine());
            else
                ChooseRandomFlyPoint();
        }
    }

    // --------------------------------------------------
    // Ground idle
    // --------------------------------------------------
    void GroundIdle()
    {
        idleTimer += Time.deltaTime;
        if (idleTimer >= idleTime)
            StartCoroutine(TakeOffRoutine());
    }

    // --------------------------------------------------
    // Waypoint selection – never pick the same point twice
    // --------------------------------------------------
    void ChooseRandomFlyPoint()
    {
        if (flyPoints.Length <= 1)
        {
            currentTarget = flyPoints[0];
            return;
        }

        Transform nextPoint;
        do
        {
            nextPoint = flyPoints[Random.Range(0, flyPoints.Length)];
        }
        while (nextPoint == currentTarget);

        currentTarget = nextPoint;
    }

    void ChooseRandomLandPoint()
    {
        if (landPoints.Length == 0) return;
        currentTarget = landPoints[Random.Range(0, landPoints.Length)];
    }

    // --------------------------------------------------
    // Landing – slower descent, matched rotation/movement
    // --------------------------------------------------
    IEnumerator LandRoutine()
    {
        isTransitioning = true;
        isFlying = false;
        ChooseRandomLandPoint();

        while (Vector3.Distance(transform.position, currentTarget.position) > pointReachDistance)
        {
            // Smooth altitude same as flying
            Vector3 targetPos = currentTarget.position;
            targetPos.y = Mathf.Lerp(transform.position.y, currentTarget.position.y, 0.3f);

            Vector3 direction = (targetPos - transform.position).normalized;

            if (direction != Vector3.zero)
            {
                Quaternion lookRot = Quaternion.LookRotation(direction);
                transform.rotation = Quaternion.Slerp(transform.rotation, lookRot, rotateSpeed * Time.deltaTime);
            }

            transform.position = Vector3.MoveTowards(
                transform.position,
                targetPos,
                landingFlySpeed * Time.deltaTime
            );

            yield return null;
        }

        // Arrived – landing animation
        anim.SetBool("FlyingFWD", false);
        anim.SetTrigger("Land");

        yield return new WaitForSeconds(2f);

        anim.SetBool("IdleSimple", true);
        idleTimer = 0f;
        isTransitioning = false;
    }

    // --------------------------------------------------
    // Takeoff
    // --------------------------------------------------
    IEnumerator TakeOffRoutine()
    {
        isTransitioning = true;

        anim.SetBool("IdleSimple", false);
        anim.SetTrigger("TakeOff");

        yield return new WaitForSeconds(2f);

        ChooseRandomFlyPoint();
        anim.SetBool("FlyingFWD", true);
        isFlying = true;
        isTransitioning = false;
    }
}