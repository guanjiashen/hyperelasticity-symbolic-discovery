subroutine luser01(nel, nuparam, nuvar, nfunc, ifunc, npf, &
                   tf, time, timestep, uparam, rho, volume, &
                   eint, ngl, soundsp, viscmax, uvar, off, &
                   sigy, pla, userbuf)
  use law_userso
  implicit none

  integer, intent(in) :: nel, nuparam, nuvar, nfunc
  integer, intent(in) :: ifunc(*), npf(*), ngl(nel)
  double precision, intent(in) :: tf(*), time, timestep
  double precision, intent(in) :: uparam(nuparam), rho(nel)
  double precision, intent(in) :: volume(nel), eint(nel)
  double precision, intent(out) :: soundsp(nel), viscmax(nel)
  double precision, intent(inout) :: uvar(nel,nuvar), off(nel)
  double precision, intent(inout) :: sigy(nel), pla(nel)
  type(ulawintbuf), intent(inout) :: userbuf

  double precision, parameter :: zero = 0.0d0
  double precision, parameter :: one = 1.0d0
  double precision, parameter :: two = 2.0d0
  double precision, parameter :: half = 0.5d0
  double precision, parameter :: third = 1.0d0 / 3.0d0
  double precision, parameter :: two_third = 2.0d0 / 3.0d0
  double precision, parameter :: em20 = 1.0d-20

  integer :: i, nterms, base
  double precision :: bulk_kappa, tensioncut, gref, iform_value
  double precision :: amat(3,3), eval(3), evec(3,3), stretch(3), l2(3)
  double precision :: sigpr(3), nominal(3)
  double precision :: jvol, rvd, i1bar, i2bar, w1, w2, pvol, trace_dev

  if (nint(uparam(1)) /= 2) then
    bulk_kappa = uparam(7)
    tensioncut = uparam(8)
    iform_value = uparam(9)
    gref = max(uparam(10), em20)
    nterms = 0
    base = 0
  else
    nterms = nint(uparam(2))
    base = 3 + 3 * nterms
    bulk_kappa = uparam(base)
    tensioncut = uparam(base + 1)
    iform_value = uparam(base + 2)
    gref = max(uparam(base + 3), em20)
  end if

  if (time == zero .and. nuvar > 0) uvar(1:nel,1:nuvar) = zero

  do i = 1, nel
    ! O-Ring uses Ismstr=0: EPS is the total logarithmic strain tensor.
    ! Radioss stores tensor shear components in engineering convention.
    amat(1,1) = userbuf%epsxx(i)
    amat(2,2) = userbuf%epsyy(i)
    amat(3,3) = userbuf%epszz(i)
    amat(1,2) = half * userbuf%epsxy(i)
    amat(2,1) = amat(1,2)
    amat(2,3) = half * userbuf%epsyz(i)
    amat(3,2) = amat(2,3)
    amat(3,1) = half * userbuf%epszx(i)
    amat(1,3) = amat(3,1)

    call law291_eigen_sym3(amat, eval, evec)
    stretch(1:3) = exp(eval(1:3))
    jvol = stretch(1) * stretch(2) * stretch(3)

    sigpr(1:3) = zero
    if (jvol > em20) then
      rvd = exp(-third * log(jvol))
      l2(1:3) = (stretch(1:3) * rvd)**2
      i1bar = l2(1) + l2(2) + l2(3)
      i2bar = l2(1)*l2(2) + l2(2)*l2(3) + l2(3)*l2(1)

      if (nterms > 0) then
        call eval_invariant_poly_derivatives(uparam, nterms, i1bar, i2bar, w1, w2)
      else
        w1 = 5.0d0 * uparam(1) * i1bar**4 + uparam(2) * i2bar**2 + uparam(3)
        w2 = two * uparam(2) * i1bar * i2bar + uparam(4)
      end if
      if (nint(iform_value) == 2) then
        pvol = bulk_kappa * (one - one / jvol)
      else
        pvol = bulk_kappa * (jvol - one)
      end if
      trace_dev = third * (i1bar*w1 + two*i2bar*w2)
      sigpr(1:3) = two * ((w1 + i1bar*w2)*l2(1:3) &
                   - w2*l2(1:3)**2 - trace_dev) / jvol + pvol

      nominal(1:3) = jvol * sigpr(1:3) / max(stretch(1:3), em20)
      if (off(i) == zero .or. maxval(nominal) > abs(tensioncut)) then
        sigpr(1:3) = zero
        off(i) = zero
      end if
    end if

    userbuf%signxx(i) = evec(1,1)**2*sigpr(1) + &
                         evec(1,2)**2*sigpr(2) + evec(1,3)**2*sigpr(3)
    userbuf%signyy(i) = evec(2,1)**2*sigpr(1) + &
                         evec(2,2)**2*sigpr(2) + evec(2,3)**2*sigpr(3)
    userbuf%signzz(i) = evec(3,1)**2*sigpr(1) + &
                         evec(3,2)**2*sigpr(2) + evec(3,3)**2*sigpr(3)
    userbuf%signxy(i) = evec(1,1)*evec(2,1)*sigpr(1) + &
                         evec(1,2)*evec(2,2)*sigpr(2) + &
                         evec(1,3)*evec(2,3)*sigpr(3)
    userbuf%signyz(i) = evec(2,1)*evec(3,1)*sigpr(1) + &
                         evec(2,2)*evec(3,2)*sigpr(2) + &
                         evec(2,3)*evec(3,3)*sigpr(3)
    userbuf%signzx(i) = evec(3,1)*evec(1,1)*sigpr(1) + &
                         evec(3,2)*evec(1,2)*sigpr(2) + &
                         evec(3,3)*evec(1,3)*sigpr(3)

    userbuf%sigvxx(i) = zero
    userbuf%sigvyy(i) = zero
    userbuf%sigvzz(i) = zero
    userbuf%sigvxy(i) = zero
    userbuf%sigvyz(i) = zero
    userbuf%sigvzx(i) = zero
    userbuf%dpla(i) = zero
    viscmax(i) = zero
    soundsp(i) = sqrt((bulk_kappa + two_third*gref) / max(rho(i), em20))

    if (nuvar >= 1) uvar(i,1) = maxval(sigpr)
    if (nuvar >= 2) uvar(i,2) = minval(sigpr)
    if (nuvar >= 3) uvar(i,3) = off(i)
  end do

end subroutine luser01


subroutine eval_invariant_poly_derivatives(uparam, nterms, i1, i2, dwd_i1, dwd_i2)
  implicit none
  integer, intent(in) :: nterms
  double precision, intent(in) :: uparam(*), i1, i2
  double precision, intent(out) :: dwd_i1, dwd_i2
  integer :: i, base, p1, p2
  double precision :: coef

  dwd_i1 = 0.0d0
  dwd_i2 = 0.0d0
  do i = 1, nterms
    base = 3 + 3 * (i - 1)
    coef = uparam(base)
    p1 = nint(uparam(base + 1))
    p2 = nint(uparam(base + 2))
    ! p1/p2 may be negative (INVARIANT_LAURENT_TERMS_V1); I1, I2 >= 3.
    if (p1 /= 0) dwd_i1 = dwd_i1 + coef * dble(p1) * i1**(p1 - 1) * i2**p2
    if (p2 /= 0) dwd_i2 = dwd_i2 + coef * dble(p2) * i1**p1 * i2**(p2 - 1)
  end do
end subroutine eval_invariant_poly_derivatives


subroutine law291_eigen_sym3(ain, eval, evec)
  implicit none
  double precision, intent(in) :: ain(3,3)
  double precision, intent(out) :: eval(3), evec(3,3)
  double precision, parameter :: one = 1.0d0
  double precision, parameter :: zero = 0.0d0
  double precision :: a(3,3), scale, tol, app, aqq, apq
  double precision :: tau, t, c, s, akp, akq, vkp, vkq
  integer :: iter, p, q, k

  a = ain
  evec = zero
  evec(1,1) = one
  evec(2,2) = one
  evec(3,3) = one
  scale = max(one, maxval(abs(a)))
  tol = 1.0d-14 * scale

  do iter = 1, 50
    p = 1
    q = 2
    if (abs(a(1,3)) > abs(a(p,q))) then
      p = 1
      q = 3
    end if
    if (abs(a(2,3)) > abs(a(p,q))) then
      p = 2
      q = 3
    end if
    if (abs(a(p,q)) <= tol) exit

    app = a(p,p)
    aqq = a(q,q)
    apq = a(p,q)
    tau = (aqq - app) / (2.0d0 * apq)
    if (tau >= zero) then
      t = one / (tau + sqrt(one + tau*tau))
    else
      t = -one / (-tau + sqrt(one + tau*tau))
    end if
    c = one / sqrt(one + t*t)
    s = t * c

    do k = 1, 3
      if (k /= p .and. k /= q) then
        akp = a(k,p)
        akq = a(k,q)
        a(k,p) = c*akp - s*akq
        a(p,k) = a(k,p)
        a(k,q) = s*akp + c*akq
        a(q,k) = a(k,q)
      end if
    end do
    a(p,p) = c*c*app - 2.0d0*s*c*apq + s*s*aqq
    a(q,q) = s*s*app + 2.0d0*s*c*apq + c*c*aqq
    a(p,q) = zero
    a(q,p) = zero

    do k = 1, 3
      vkp = evec(k,p)
      vkq = evec(k,q)
      evec(k,p) = c*vkp - s*vkq
      evec(k,q) = s*vkp + c*vkq
    end do
  end do

  eval(1) = a(1,1)
  eval(2) = a(2,2)
  eval(3) = a(3,3)
end subroutine law291_eigen_sym3
