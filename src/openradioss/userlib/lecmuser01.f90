subroutine lecmuser01(iin, iout, uparam, maxuparam, nuparam, &
                      nuvar, ifunc, maxfunc, nfunc, stifint, userbuf)
  use law_user
  implicit none

  integer, intent(in) :: iin, iout, maxuparam, maxfunc
  integer, intent(out) :: nuparam, nuvar, nfunc
  integer, intent(inout) :: ifunc(maxfunc)
  double precision, intent(out) :: uparam(maxuparam), stifint
  type(ulawbuf), intent(inout) :: userbuf

  double precision, parameter :: zero = 0.0d0
  double precision, parameter :: two = 2.0d0
  double precision, parameter :: three = 3.0d0
  double precision, parameter :: em20 = 1.0d-20
  double precision, parameter :: ep20 = 1.0d20
  integer, parameter :: max_terms = 20
  double precision :: acoef, bcoef, ccoef, dcoef, eshift, freserved
  double precision :: nu, sigcut, dwd_i1_ref, dwd_i2_ref
  double precision :: gref, bulk_kappa, young, package_gref, rmse
  double precision :: term_coef(max_terms)
  integer :: iform, nterms, term_i1(max_terms), term_i2(max_terms), base
  character(len=512) :: line, mode, package_path, cache_path
  character(len=1024) :: command, python_cmd, exporter
  character(len=80) :: flat_header, expr_type, model_hash
  character(len=64) :: real_to_text
  character(len=1024) :: shell_quote
  integer :: ios, exitstat
  double precision :: i1min, i1max, i2min, i2max

  if (maxuparam < 80) then
    write(iout,*) ' ** ERROR: LAW291 USER01 requires at least 80 UPARAM entries'
    call arret(1)
  end if

  ! The density line of /MAT/USER01 is consumed by the generic reader.
  ! Legacy cards provide coefficients directly; new cards start with a mode.
  read(iin,'(A)',iostat=ios) line
  if (ios /= 0) then
    write(iout,*) ' ** ERROR: LAW291 USER01 could not read material data'
    call arret(1)
  end if
  mode = adjustl(line)

  if (mode(1:7) == 'PACKAGE') then
    read(iin,'(A)') package_path
    read(iin,*) nu, sigcut
    read(iin,*) iform
    call read_nn_flat_package(trim(adjustl(package_path)), iout, nterms, term_coef, &
         term_i1, term_i2, max_terms, bulk_kappa, package_gref, &
         i1min, i1max, i2min, i2max, rmse, flat_header, expr_type, model_hash)
  else if (mode(1:5) == 'TRAIN') then
    read(iin,'(A)') package_path
    read(iin,'(A)') cache_path
    read(iin,*) nu, sigcut
    read(iin,*) iform
    call get_environment_variable('RAD_NN_PYTHON', python_cmd, status=ios)
    if (ios /= 0 .or. len_trim(python_cmd) == 0) python_cmd = 'python3'
    call get_environment_variable('RAD_NN_EXPORTER', exporter, status=ios)
    if (ios /= 0 .or. len_trim(exporter) == 0) &
      exporter = 'tools/nn_invariant/export_openradioss.py'
    cache_path = trim(adjustl(cache_path)) // '/material.flat'
    write(command,'(A)') trim(shell_quote(trim(python_cmd))) // ' ' // &
      trim(shell_quote(trim(exporter))) // &
      ' export-openradioss --config ' // trim(shell_quote(trim(adjustl(package_path)))) // &
      ' --output ' // trim(shell_quote(trim(adjustl(cache_path)))) // &
      ' --nu ' // trim(real_to_text(nu))
    write(iout,'(/,5x,a)') 'NN_INVARIANT: running export-openradioss'
    write(iout,'(5x,a,a)') 'Command: ', trim(command)
    call execute_command_line(trim(command), exitstat=exitstat)
    if (exitstat /= 0) then
      write(iout,*) ' ** ERROR: NN_invariant export-openradioss failed'
      call arret(1)
    end if
    call read_nn_flat_package(trim(adjustl(cache_path)), iout, nterms, term_coef, &
         term_i1, term_i2, max_terms, bulk_kappa, package_gref, &
         i1min, i1max, i2min, i2max, rmse, flat_header, expr_type, model_hash)
  else
    read(line,*) acoef, bcoef, ccoef
    read(iin,*) dcoef, eshift, freserved
    read(iin,*) nu, sigcut
    read(iin,*) iform
    package_gref = zero
    i1min = zero
    i1max = zero
    i2min = zero
    i2max = zero
    rmse = zero
    flat_header = 'LEGACY_COEFFICIENTS'
    expr_type = 'LAW291_POLYNOMIAL_V1'
    model_hash = 'manual'
    bulk_kappa = zero
    nterms = 4
    term_coef(1) = acoef
    term_i1(1) = 5
    term_i2(1) = 0
    term_coef(2) = bcoef
    term_i1(2) = 1
    term_i2(2) = 2
    term_coef(3) = ccoef
    term_i1(3) = 1
    term_i2(3) = 0
    term_coef(4) = dcoef
    term_i1(4) = 0
    term_i2(4) = 1
  end if

  if (nu == zero) nu = 0.495d0
  if (nu < zero .or. nu >= 0.5d0) then
    write(iout,*) ' ** ERROR: LAW291 USER01 requires 0 <= NU < 0.5'
    call arret(1)
  end if
  if (sigcut <= zero) sigcut = ep20
  if (iform <= 0) iform = 1
  iform = min(2, iform)

  ! Initial invariant derivatives at I1bar=I2bar=3.
  call eval_poly_derivatives(nterms, term_coef, term_i1, term_i2, three, three, dwd_i1_ref, dwd_i2_ref)
  gref = two * (dwd_i1_ref + dwd_i2_ref)
  if (gref <= zero) then
    write(iout,*) ' ** ERROR: LAW291 USER01 initial shear modulus is not positive'
    call arret(1)
  end if

  ! Same bulk-modulus definition used by native LAW291 and LAW42.
  if (bulk_kappa <= zero) bulk_kappa = two * gref * (1.0d0 + nu) / &
                                      max(em20, three * (1.0d0 - two * nu))
  young = two * gref * (1.0d0 + nu)

  uparam(:) = zero
  uparam(1) = 2.0d0
  uparam(2) = dble(nterms)
  do ios = 1, nterms
    base = 3 + 3 * (ios - 1)
    uparam(base) = term_coef(ios)
    uparam(base + 1) = dble(term_i1(ios))
    uparam(base + 2) = dble(term_i2(ios))
  end do
  base = 3 + 3 * nterms
  uparam(base) = bulk_kappa
  uparam(base + 1) = sigcut
  uparam(base + 2) = dble(iform)
  uparam(base + 3) = gref
  uparam(base + 4) = nu
  uparam(base + 5) = i1min
  uparam(base + 6) = i1max
  uparam(base + 7) = i2min
  uparam(base + 8) = i2max
  uparam(base + 9) = rmse

  nuparam = base + 9
  nuvar = 3
  nfunc = 0
  if (maxfunc > 0) ifunc(1) = 0

  ! Longitudinal stiffness used by interfaces/contact estimates.
  stifint = bulk_kappa + 4.0d0 * gref / 3.0d0

  write(iout,'(/,5x,a)') 'USER01: invariant polynomial hyperelastic LAW291'
  write(iout,'(5x,a,a)') 'Package format        = ', trim(flat_header)
  write(iout,'(5x,a,a)') 'Expression type       = ', trim(expr_type)
  write(iout,'(5x,a,a)') 'Model hash            = ', trim(model_hash)
  write(iout,'(5x,a,i4)') 'Number of terms       = ', nterms
  do ios = 1, nterms
    write(iout,'(7x,a,i2,a,1pe16.8,a,i3,a,i3)') 'term ', ios, ': coef = ', term_coef(ios), &
      '  I1^', term_i1(ios), '  I2^', term_i2(ios)
  end do
  if (rmse > zero) write(iout,'(5x,a,1pe12.4)') 'SR fit RMSE           = ', rmse
  write(iout,'(5x,a,1pe12.4)') 'Initial shear modulus = ', gref
  write(iout,'(5x,a,1pe12.4)') 'Bulk modulus          = ', bulk_kappa
  write(iout,'(5x,a,1pe12.4)') 'Young modulus         = ', young
  write(iout,'(5x,a,0pf10.6,/)') 'Poisson ratio         = ', nu

end subroutine lecmuser01


subroutine read_nn_flat_package(path, iout, nterms, term_coef, term_i1, term_i2, max_terms, &
                                bulk_kappa, gref, &
                                i1min, i1max, i2min, i2max, rmse, &
                                flat_header, expr_type, model_hash)
  implicit none
  character(len=*), intent(in) :: path
  integer, intent(in) :: iout, max_terms
  integer, intent(out) :: nterms, term_i1(max_terms), term_i2(max_terms)
  double precision, intent(out) :: term_coef(max_terms)
  double precision, intent(out) :: bulk_kappa, gref
  double precision, intent(out) :: i1min, i1max, i2min, i2max, rmse
  character(len=*), intent(out) :: flat_header, expr_type, model_hash
  integer :: unit, ios, i

  open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(iout,*) ' ** ERROR: cannot open NN_invariant material package: ', trim(path)
    call arret(1)
  end if
  read(unit,'(A)',iostat=ios) flat_header
  if (ios /= 0 .or. trim(flat_header) /= 'NN_INVARIANT_MATERIAL_V1') then
    write(iout,*) ' ** ERROR: unsupported NN_invariant material package'
    call arret(1)
  end if
  read(unit,'(A)') expr_type
  ! INVARIANT_POLYNOMIAL_TERMS_V1: coef * I1**p1 * I2**p2 with p1, p2 >= 0.
  ! INVARIANT_LAURENT_TERMS_V1   : same layout, p1/p2 may be negative
  !   (produced when PySR uses '/' or when post-fit pruning keeps 1/I2 terms).
  if (trim(expr_type) /= 'INVARIANT_POLYNOMIAL_TERMS_V1' .and. &
      trim(expr_type) /= 'INVARIANT_LAURENT_TERMS_V1') then
    write(iout,*) ' ** ERROR: unsupported NN_invariant expression type: ', trim(expr_type)
    call arret(1)
  end if
  read(unit,*) nterms
  if (nterms <= 0 .or. nterms > max_terms) then
    write(iout,*) ' ** ERROR: invalid NN_invariant polynomial term count: ', nterms
    call arret(1)
  end if
  term_coef(:) = 0.0d0
  term_i1(:) = 0
  term_i2(:) = 0
  do i = 1, nterms
    read(unit,*,iostat=ios) term_coef(i), term_i1(i), term_i2(i)
    if (ios /= 0) then
      write(iout,*) ' ** ERROR: cannot read NN_invariant term ', i, ' from ', trim(path)
      call arret(1)
    end if
    if (trim(expr_type) == 'INVARIANT_POLYNOMIAL_TERMS_V1' .and. &
        (term_i1(i) < 0 .or. term_i2(i) < 0)) then
      write(iout,*) ' ** ERROR: negative exponent in INVARIANT_POLYNOMIAL_TERMS_V1 term ', i
      call arret(1)
    end if
  end do
  read(unit,*) bulk_kappa
  read(unit,*) gref
  read(unit,*) i1min, i1max
  read(unit,*) i2min, i2max
  read(unit,*) rmse
  read(unit,'(A)') model_hash
  close(unit)
end subroutine read_nn_flat_package


subroutine eval_poly_derivatives(nterms, term_coef, term_i1, term_i2, i1, i2, dwd_i1, dwd_i2)
  implicit none
  integer, intent(in) :: nterms, term_i1(*), term_i2(*)
  double precision, intent(in) :: term_coef(*), i1, i2
  double precision, intent(out) :: dwd_i1, dwd_i2
  integer :: i, p1, p2

  ! Integer exponents may be negative (Laurent terms); I1, I2 >= 3 for the
  ! isochoric invariants, so negative powers are always well defined.
  dwd_i1 = 0.0d0
  dwd_i2 = 0.0d0
  do i = 1, nterms
    p1 = term_i1(i)
    p2 = term_i2(i)
    if (p1 /= 0) dwd_i1 = dwd_i1 + term_coef(i) * dble(p1) * i1**(p1 - 1) * i2**p2
    if (p2 /= 0) dwd_i2 = dwd_i2 + term_coef(i) * dble(p2) * i1**p1 * i2**(p2 - 1)
  end do
end subroutine eval_poly_derivatives


function real_to_text(value) result(text)
  implicit none
  double precision, intent(in) :: value
  character(len=64) :: text
  write(text,'(1pe24.16)') value
end function real_to_text


function shell_quote(value) result(quoted)
  implicit none
  character(len=*), intent(in) :: value
  character(len=1024) :: quoted
  integer :: i, pos, n

  quoted = ' '
  quoted(1:1) = "'"
  pos = 2
  n = len_trim(value)
  do i = 1, n
    if (value(i:i) == "'") then
      if (pos + 3 > len(quoted)) exit
      quoted(pos:pos+3) = "'\''"
      pos = pos + 4
    else
      if (pos > len(quoted)) exit
      quoted(pos:pos) = value(i:i)
      pos = pos + 1
    end if
  end do
  if (pos <= len(quoted)) quoted(pos:pos) = "'"
end function shell_quote
